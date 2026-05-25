# Security Policy

## Supported versions

Slate is in alpha. Only the `main` branch is supported for security updates.

## Reporting a vulnerability

Please report security vulnerabilities **privately** so we can address them before
public disclosure.

- **Preferred**: open a private security advisory on GitHub (Security → Advisories →
  Report a vulnerability).
- **Alternative**: email the maintainers at the address listed in the repository's
  GitHub `About` section.

Please include:

- A description of the vulnerability and its impact
- Steps to reproduce
- Affected version (commit hash)
- Any proof-of-concept code

We will acknowledge receipt within 7 days and aim to provide a fix or mitigation within
30 days for high-severity issues.

## Threat model and explicit non-guarantees

Slate exists in a context with several inherent risks. We are honest about which we
defend against and which we do not.

### What Slate defends against

- **Untrusted model outputs executed as code.** The `CodeExecutor` interface and its
  default Wasmtime+Pyodide implementation are designed to safely execute code generated
  by language models. Sandbox escape via the default executor is in-scope.
- **Untrusted training data parsing.** Tokenizer, dataset loader, and GGUF parsers must
  not be exploitable via crafted inputs. Crashes on adversarial inputs are in-scope.
- **Crashes that lead to file corruption.** Checkpoint writes are atomic
  (write-tmp + rename); a partial write must not corrupt the active state.

### What Slate does **not** defend against

- **Compromised toolchains or supply chain.** We do not pin or verify compiler
  binaries, build systems, or third-party tools beyond what CMake and your OS package
  manager provide.
- **Side-channel attacks** (timing, power, electromagnetic). Slate is not designed for
  multi-tenant or hostile co-tenant environments.
- **Malicious model weights.** If you load weights you did not produce or verify, you
  should assume they may behave adversarially. We do not detect this.
- **Physical or root-level attackers.** An attacker with kernel-level access on the
  training machine is out of scope.

### Important caveats for the `CodeExecutor`

The `WasmtimePyodideExecutor` runs Python code inside a WebAssembly sandbox with no
network and no filesystem access. This is significantly safer than running code
directly, but is not formally verified. Do not rely on it for:

- Hosting code execution as a service to untrusted users
- Sandboxing code from sources with capability over you (state-level adversaries, etc.)

For training your own model on a single-user device, the threat model is "an LLM
generates code that might accidentally do something harmful" — which the default
sandbox handles well.

## Disclosure policy

We follow a **coordinated disclosure** model:

1. Vulnerability reported privately.
2. We confirm and develop a fix.
3. We coordinate a disclosure date with the reporter (typically 30–90 days from
   confirmation).
4. We release the fix, then publish a security advisory naming the reporter (with
   their consent).

Credit will be given to reporters who follow this process.
