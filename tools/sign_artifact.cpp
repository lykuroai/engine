// Development signing tool (release-pipeline stand-in).
//
//   sign_artifact keygen <out_prefix>
//       Writes <out_prefix>.pub (hex public key) and <out_prefix>.key
//       (hex seed||pub private key, keep secret).
//   sign_artifact sign <private_key_file> <artifact_dir>
//       Signs <artifact_dir>/manifest.json into <artifact_dir>/manifest.sig.

#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>

#include "security/signature.h"

namespace {

using namespace lykuro::nie;

std::string ReadAll(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

std::string Trim(std::string s) {
    while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) s.pop_back();
    return s;
}

int Keygen(const std::string& prefix) {
    std::array<uint8_t, 32> pub;
    std::array<uint8_t, 64> priv;
    Status s = GenerateKeypair(pub, priv);
    if (!s.ok()) {
        std::fprintf(stderr, "keygen failed: %s\n", s.message().c_str());
        return 1;
    }
    std::ofstream(prefix + ".pub") << BytesToHex(pub.data(), pub.size())
                                   << "\n";
    std::ofstream(prefix + ".key") << BytesToHex(priv.data(), priv.size())
                                   << "\n";
    std::printf("wrote %s.pub and %s.key\n", prefix.c_str(), prefix.c_str());
    return 0;
}

int Sign(const std::string& key_path, const std::string& artifact_dir) {
    std::array<uint8_t, 64> priv;
    Status s = HexToBytes(Trim(ReadAll(key_path)), priv.data(), priv.size());
    if (!s.ok()) {
        std::fprintf(stderr, "invalid private key file\n");
        return 1;
    }
    std::string manifest = ReadAll(artifact_dir + "/manifest.json");
    if (manifest.empty()) {
        std::fprintf(stderr, "manifest.json not found or empty\n");
        return 1;
    }
    std::array<uint8_t, 64> sig;
    s = SignMessage(manifest, priv, sig);
    if (!s.ok()) {
        std::fprintf(stderr, "signing failed: %s\n", s.message().c_str());
        return 1;
    }
    std::ofstream(artifact_dir + "/manifest.sig")
        << BytesToHex(sig.data(), sig.size()) << "\n";
    std::printf("wrote %s/manifest.sig\n", artifact_dir.c_str());
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc == 3 && std::strcmp(argv[1], "keygen") == 0) {
        return Keygen(argv[2]);
    }
    if (argc == 4 && std::strcmp(argv[1], "sign") == 0) {
        return Sign(argv[2], argv[3]);
    }
    std::fprintf(stderr,
                 "usage:\n"
                 "  sign_artifact keygen <out_prefix>\n"
                 "  sign_artifact sign <private_key_file> <artifact_dir>\n");
    return 2;
}
