//! Useful functions for checking features.
use std::str::FromStr;

pub fn get_num_threads() -> usize {
    // Respond to the same environment variable as rayon.
    match std::env::var("RAYON_NUM_THREADS")
        .ok()
        .and_then(|s| usize::from_str(&s).ok())
    {
        Some(x) if x > 0 => x,
        Some(_) | None => num_cpus::get(),
    }
}

/// Returns the appropriate `gemm::Parallelism` setting, avoiding nested Rayon
/// parallelism. When already executing on a Rayon worker thread (i.e., inside
/// a `par_iter` or `rayon::scope`), requesting `Parallelism::Rayon(N)` from
/// the `gemm` crate causes threadpool starvation: the outer Rayon tasks hold
/// worker threads while the inner gemm tasks queue behind them, resulting in
/// single-threaded execution.  Detecting this via `rayon::current_thread_index()`
/// and falling back to `Parallelism::None` lets the calling worker do the gemm
/// work directly, while sibling workers handle their own gemm calls in parallel.
pub fn get_gemm_parallelism() -> gemm::Parallelism {
    // If we're already on a Rayon worker thread, don't nest.
    if rayon::current_thread_index().is_some() {
        return gemm::Parallelism::None;
    }
    let num_threads = get_num_threads();
    if num_threads > 1 {
        gemm::Parallelism::Rayon(num_threads)
    } else {
        gemm::Parallelism::None
    }
}

pub fn has_accelerate() -> bool {
    cfg!(feature = "accelerate")
}

pub fn has_mkl() -> bool {
    cfg!(feature = "mkl")
}

pub fn cuda_is_available() -> bool {
    cfg!(feature = "cuda")
}

pub fn metal_is_available() -> bool {
    cfg!(feature = "metal")
}

pub fn with_avx() -> bool {
    cfg!(target_feature = "avx2")
}

pub fn with_neon() -> bool {
    cfg!(target_feature = "neon")
}

pub fn with_simd128() -> bool {
    cfg!(target_feature = "simd128")
}

pub fn with_f16c() -> bool {
    cfg!(target_feature = "f16c")
}
