#!/usr/bin/env bash
# CVE gate for the Lykuro Native Inference Engine.
#
# Reads the SBOM (SPDX) and the VEX review ledger, cross-checks each
# OSV-gated dependency against api.osv.dev, and writes a machine-readable
# vulnerability-report.json. Fail-closed on inventory: every SBOM package
# must be accounted for in the ledger. Gate policy (severities that block)
# comes from the ledger's policy.gate_severity.
#
# Exit 0 = gate PASS, 1 = gate FAIL, 2 = usage/tooling error.
#
# Deps: bash, jq, curl (CI tooling only; never linked into the engine).
# Offline mode (--offline) skips OSV and still enforces the inventory and
# expired-waiver checks, marking OSV findings as "not_evaluated".
set -euo pipefail

SBOM="sbom/lykuro-native-engine.spdx.json"
LEDGER="security/vex-ledger.json"
OUT="vulnerability-report.json"
ONLINE=1
TODAY="${SCAN_TODAY:-$(date -u +%Y-%m-%d)}"
SEP=$'\t'

usage() { echo "usage: $0 [--sbom F] [--ledger F] [--report F] [--offline]"; exit 2; }
while [ $# -gt 0 ]; do
  case "$1" in
    --sbom) SBOM="$2"; shift 2;;
    --ledger) LEDGER="$2"; shift 2;;
    --report|--out) OUT="$2"; shift 2;;
    --offline) ONLINE=0; shift;;
    -h|--help) usage;;
    *) echo "unknown arg: $1" >&2; usage;;
  esac
done

command -v jq >/dev/null 2>&1 || { echo "FATAL: jq not found" >&2; exit 2; }
command -v curl >/dev/null 2>&1 || { echo "FATAL: curl not found" >&2; exit 2; }
[ -f "$SBOM" ] || { echo "FATAL: SBOM not found: $SBOM" >&2; exit 2; }
[ -f "$LEDGER" ] || { echo "FATAL: ledger not found: $LEDGER" >&2; exit 2; }

OSV_EP="$(jq -r '.policy.osv_endpoint // "https://api.osv.dev"' "$LEDGER")"
GATE_SEV="$(jq -c '.policy.gate_severity' "$LEDGER")"

TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT

# normalize a raw OSV severity string to CRITICAL/HIGH/MEDIUM/LOW/UNKNOWN
norm_sev() {
  case "$(printf '%s' "$1" | tr '[:lower:]' '[:upper:]')" in
    CRITICAL) echo CRITICAL;;
    HIGH) echo HIGH;;
    MODERATE|MEDIUM) echo MEDIUM;;
    LOW|NEGLIGIBLE) echo LOW;;
    *) echo UNKNOWN;;
  esac
}

# fetch one OSV vuln, print "<SEVERITY><TAB><cvss-or-->"; caches by id
sev_of() {
  local id="$1" f="$TMP/vuln_$1.json" raw cvss s
  [ -s "$f" ] || curl -sS --max-time 15 "$OSV_EP/v1/vulns/$id" >"$f" 2>/dev/null || true
  raw="$(jq -r '.database_specific.severity // ([.severity[]?|select(.type=="Ubuntu")|.score]|first) // ""' "$f" 2>/dev/null || true)"
  cvss="$(jq -r '[.severity[]?|select(.type|startswith("CVSS"))|.score]|first // "-"' "$f" 2>/dev/null || true)"
  s="$(norm_sev "$raw")"
  printf '%s%s%s\n' "$s" "$SEP" "$cvss"
}

# SBOM packages -> "name<TAB>version" lines
PKGS=()
while IFS= read -r line; do PKGS+=("$line"); done \
  < <(jq -r --arg s "$SEP" '.packages[] | "\(.name)\($s)\(.versionInfo)"' "$SBOM")

FAIL=0
REASONS="$TMP/reasons.txt"; : > "$REASONS"
COMPS="$TMP/comps.jsonl"; : > "$COMPS"

for row in "${PKGS[@]}"; do
  name="${row%%${SEP}*}"; version="${row#*${SEP}}"
  entry="$(jq -c --arg n "$name" '.components[] | select(.sbom_name==$n)' "$LEDGER")"
  if [ -z "$entry" ]; then
    echo "unreviewed_dependency: $name@$version has no ledger entry" >>"$REASONS"
    FAIL=1
    jq -cn --arg n "$name" --arg v "$version" \
      '{sbom_name:$n, version:$v, disposition:"UNREVIEWED", gated:false, justification:"", findings:[], blocking:[]}' >>"$COMPS"
    continue
  fi
  disp="$(jq -r '.disposition' <<<"$entry")"
  osv="$(jq -c '.osv // empty' <<<"$entry")"

  findings='[]'; blocking='[]'
  if [ -n "$osv" ] && [ "$ONLINE" -eq 1 ]; then
    oname="$(jq -r '.name' <<<"$osv")"; oeco="$(jq -r '.ecosystem' <<<"$osv")"; over="$(jq -r '.version' <<<"$osv")"
    resp="$(curl -sS --max-time 20 -X POST -H 'Content-Type: application/json' \
      -d "$(jq -cn --arg n "$oname" --arg e "$oeco" --arg v "$over" \
            '{queries:[{package:{name:$n,ecosystem:$e},version:$v}]}')" \
      "$OSV_EP/v1/querybatch" 2>/dev/null || echo '{}')"
    IDS=()
    while IFS= read -r id; do [ -n "$id" ] && IDS+=("$id"); done \
      < <(jq -r '.results[0].vulns[]?.id' <<<"$resp" 2>/dev/null | grep -E '^(UBUNTU-CVE|CVE|GHSA|GO|PYSEC)' || true)
    fbuf="$TMP/find.jsonl"; : > "$fbuf"
    for id in "${IDS[@]}"; do
      sev=""; cvss=""
      IFS="$SEP" read -r sev cvss < <(sev_of "$id") || true
      wv="$(jq -c --arg id "$id" '(.waivers // [])[] | select(.id==$id or .id=="*")' <<<"$entry" | head -1)"
      waived=false; wstatus=""; wdue=""; wjust=""
      if [ -n "$wv" ]; then
        waived=true
        wstatus="$(jq -r '.status // ""' <<<"$wv")"
        wdue="$(jq -r '.due // ""' <<<"$wv")"
        wjust="$(jq -r '.exposure // .action // ""' <<<"$wv")"
        if [ -n "$wdue" ] && [ "$wdue" \< "$TODAY" ]; then
          echo "expired_waiver: $name $id waiver due $wdue < $TODAY" >>"$REASONS"
          FAIL=1; waived=false
        fi
      fi
      gated=false
      if jq -e --arg s "$sev" 'index($s)' <<<"$GATE_SEV" >/dev/null; then gated=true; fi
      blocks=false
      if [ "$gated" = true ] && [ "$waived" != true ]; then
        blocks=true; FAIL=1
        echo "untriaged_gated_finding: $name $id ($sev)" >>"$REASONS"
      fi
      jq -cn --arg id "$id" --arg s "$sev" --arg c "$cvss" \
        --argjson w "$waived" --arg ws "$wstatus" --arg wd "$wdue" --arg wj "$wjust" \
        --argjson g "$gated" --argjson b "$blocks" \
        '{id:$id, severity:$s, cvss:$c, gated:$g, waived:$w, waiver_status:$ws, waiver_due:$wd, note:$wj, blocking:$b}' >>"$fbuf"
    done
    findings="$(jq -cs '.' "$fbuf")"
    blocking="$(jq -c '[.[]|select(.blocking)|.id]' <<<"$findings")"
  elif [ -n "$osv" ] && [ "$ONLINE" -eq 0 ]; then
    echo "note: OSV not evaluated (offline) for $name" >>"$REASONS"
  fi

  gatedflag=false; [ -n "$osv" ] && gatedflag=true
  jq -cn --arg n "$name" --arg v "$version" --arg d "$disp" \
    --argjson gated "$gatedflag" \
    --argjson findings "$findings" --argjson blocking "$blocking" \
    --arg just "$(jq -r '.justification // ""' <<<"$entry")" \
    '{sbom_name:$n, version:$v, disposition:$d, gated:$gated, justification:$just, findings:$findings, blocking:$blocking}' >>"$COMPS"
done

GENERATED="${TODAY}T00:00:00Z"
jq -s \
  --arg gen "$GENERATED" --arg sbom "$SBOM" --arg ledger "$LEDGER" \
  --argjson online "$([ "$ONLINE" -eq 1 ] && echo true || echo false)" \
  --argjson gate "$GATE_SEV" \
  --argjson failed "$([ "$FAIL" -eq 0 ] && echo false || echo true)" \
  --rawfile reasons "$REASONS" \
  '{
     schema: "lykuro.vulnerability-report/1",
     generated: $gen,
     sbom: $sbom,
     ledger: $ledger,
     osv_evaluated: $online,
     gate_policy: {block_severities: $gate},
     components: .,
     summary: {
       components_total: (.|length),
       unreviewed: ([.[]|select(.disposition=="UNREVIEWED")]|length),
       findings_total: ([.[].findings[]?]|length),
       by_severity: (reduce (.[].findings[]?) as $f ({}; .[$f.severity] = ((.[$f.severity]//0)+1))),
       blocking_total: ([.[].blocking[]?]|length),
       blocking_ids: [.[].blocking[]?]
     },
     requires_security_signoff: [
       .[] | select((.findings|length)>0) |
       {component:.sbom_name, waived_gated:[.findings[]|select(.gated and .waived)|{id,severity,waiver_status,waiver_due,note}]}
       | select(.waived_gated|length>0)
     ],
     gate: { failed: $failed, reasons: ($reasons | split("\n") | map(select(length>0))) }
   }' "$COMPS" > "$OUT"

echo "wrote $OUT"
jq -r '"components=\(.summary.components_total) findings=\(.summary.findings_total) blocking=\(.summary.blocking_total) by_severity=\(.summary.by_severity)"' "$OUT"
if [ "$FAIL" -ne 0 ]; then
  echo "GATE: FAIL"; jq -r '.gate.reasons[]' "$OUT" | sed 's/^/  - /'
  exit 1
fi
echo "GATE: PASS"
exit 0
