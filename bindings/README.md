# Bindings

This directory contains the language-specific Mesh SDK bindings built on top of
the shared native client core.

Current bindings:

- `swift/` for Apple platforms
- `kotlin/` for Android and JVM consumers

These bindings should stay thin. Shared client behavior belongs in the Rust SDK
crates:

- `mesh-client/` for the low-level client implementation
- `mesh-api/` for the public Rust client API
- `mesh-api-ffi/` for the UniFFI/native bridge used by language bindings

If you add another top-level binding here, include a `README.md` in that
binding directory explaining its packaging and public surface.
