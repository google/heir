use emit_poulpy_compiles::setup;
use poulpy_cpu_portable::{FFT64Portable, NTT4x30Portable};

#[test]
fn setup_runs_on_portable_backends() {
    setup::<FFT64Portable>().unwrap();
    setup::<NTT4x30Portable>().unwrap();
}
