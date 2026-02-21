// RUN: heir-opt --annotate-module="backend=invalid_backend" --verify-diagnostics %s

// expected-error @+1 {{Unknown backend: invalid_backend}}
module {
}