---
title: ML with HEIR
linkTitle: ML with HEIR
weight: 12
description: >-
  HEIR's ML frontend, compilation pipeline, hardware integrations, and active
  research directions.
---

<script>
  document.addEventListener("DOMContentLoaded", function () {
    document.body.classList.add("mlheir-page-body");
  });
</script>

<style>
  body:has(.mlheir-hero) .td-sidebar-toc,
  body.mlheir-page-body .td-sidebar-toc,
  body:has(.mlheir-hero) .td-breadcrumbs,
  body.mlheir-page-body .td-breadcrumbs,
  body:has(.mlheir-hero) .td-content > h1:first-child,
  body.mlheir-page-body .td-content > h1:first-child,
  body:has(.mlheir-hero) .td-content > .lead,
  body.mlheir-page-body .td-content > .lead,
  body:has(.mlheir-hero) .td-content > header.article-meta,
  body.mlheir-page-body .td-content > header.article-meta {
    display: none !important;
  }

  body:has(.mlheir-hero) .td-main main.col-xl-8,
  body.mlheir-page-body .td-main main.col-xl-8 {
    flex: 0 0 83.333333%;
    max-width: 83.333333%;
  }

  body:has(.mlheir-hero) .td-content,
  body.mlheir-page-body .td-content {
    max-width: none;
  }

  .td-content .mlheir {
    --mlheir-bg: #f4f3ef;
    --mlheir-card: #ffffff;
    --mlheir-ink: #16181d;
    --mlheir-muted: #636973;
    --mlheir-blue: #4a80ec;
    --mlheir-gold: #e4b54c;
    --mlheir-red: #e15b48;
    --mlheir-green: #4ba269;
    --mlheir-shadow: 0 16px 36px rgba(22, 24, 29, 0.08);
    color: var(--mlheir-ink);
  }

  .td-content .mlheir a {
    color: #0f7daa;
  }

  .td-content .mlheir p,
  .td-content .mlheir li {
    color: var(--mlheir-muted);
    line-height: 1.5;
  }

  .td-content .mlheir h2,
  .td-content .mlheir h3,
  .td-content .mlheir h4 {
    color: var(--mlheir-ink);
    margin-top: 0;
  }

  .td-content .mlheir-hero {
    display: grid;
    gap: 1.5rem;
    grid-template-columns: minmax(0, 1.55fr) minmax(280px, 0.95fr);
    align-items: center;
    padding: 1.75rem 1.75rem 1.4rem;
    background: linear-gradient(180deg, #f7f5f1 0%, #f2f1ed 100%);
    border-top: 4px solid var(--mlheir-ink);
    border-bottom: 5px solid var(--mlheir-blue);
    border-radius: 1.25rem;
    box-shadow: var(--mlheir-shadow);
    margin-bottom: 1.75rem;
  }

  .td-content .mlheir-hero h1 {
    font-size: clamp(2.45rem, 5vw, 4.5rem);
    line-height: 0.96;
    letter-spacing: -0.06em;
    margin: 0 0 1rem;
  }

  .td-content .mlheir-lede {
    font-size: 1.2rem;
    max-width: 36rem;
    margin: 0 0 1rem;
  }

  .td-content .mlheir-meta {
    display: flex;
    flex-wrap: wrap;
    gap: 0.75rem;
    margin: 0;
  }

  .td-content .mlheir-meta a,
  .td-content .mlheir-meta span {
    display: inline-flex;
    align-items: center;
    padding: 0.42rem 0.8rem;
    border-radius: 999px;
    background: rgba(74, 128, 236, 0.09);
    color: #1f5ad3;
    font-size: 0.93rem;
    text-decoration: none;
  }

  .td-content .mlheir-hero-side {
    display: grid;
    gap: 1rem;
    justify-items: end;
  }

  .td-content .mlheir-hero-side img {
    width: min(100%, 240px);
    height: auto;
  }

  .td-content .mlheir-side-note {
    max-width: 21rem;
    justify-self: stretch;
    padding: 1rem 1.1rem;
    background: rgba(255, 255, 255, 0.72);
    border: 1px solid rgba(22, 24, 29, 0.08);
    border-radius: 1rem;
    box-shadow: 0 8px 24px rgba(22, 24, 29, 0.05);
  }

  .td-content .mlheir-side-note p {
    margin: 0;
  }

  .td-content .mlheir-band {
    margin: 2rem 0 1.2rem;
    padding-top: 0.75rem;
    border-top: 5px solid var(--mlheir-blue);
  }

  .td-content .mlheir h2.mlheir-band-heading {
    display: inline-flex;
    align-items: center;
    gap: 0.55rem;
    min-width: 12rem;
    padding: 0.52rem 1.15rem;
    border-radius: 999px;
    background: #050607;
    color: #fff;
    font-size: 1rem;
    font-weight: 700;
    letter-spacing: 0.01em;
    text-align: center;
    margin: 0;
    scroll-margin-top: 5rem;
    color: #fff;
  }

  .td-content .mlheir-band-gold {
    border-top-color: var(--mlheir-gold);
  }

  .td-content .mlheir-band-red {
    border-top-color: var(--mlheir-red);
  }

  .td-content .mlheir-band-green {
    border-top-color: var(--mlheir-green);
  }

  .td-content .mlheir-grid {
    display: grid;
    gap: 1.25rem;
    margin: 1rem 0 1.5rem;
  }

  .td-content .mlheir-band-anchor {
    color: rgba(255, 255, 255, 0.75);
    font-size: 0.92rem;
    text-decoration: none;
    opacity: 0;
    transition: opacity 120ms ease;
  }

  .td-content .mlheir-band-heading:hover .mlheir-band-anchor,
  .td-content .mlheir-band-heading:focus-within .mlheir-band-anchor {
    opacity: 1;
  }

  .td-content .mlheir-band-anchor:hover {
    color: #fff;
  }

  .td-content .mlheir-grid-two {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }

  .td-content .mlheir-grid-three {
    grid-template-columns: repeat(3, minmax(0, 1fr));
  }

  .td-content .mlheir-grid-four {
    grid-template-columns: repeat(4, minmax(0, 1fr));
  }

  .td-content .mlheir-grid-hardware {
    grid-template-columns: minmax(260px, 0.8fr) minmax(0, 1.4fr);
    align-items: start;
  }

  .td-content .mlheir-layout-row {
    grid-template-columns: minmax(0, 1.5fr) minmax(280px, 0.9fr);
    align-items: start;
  }

  .td-content .mlheir-vendor-grid {
    grid-template-columns: repeat(3, minmax(0, 1fr));
  }

  .td-content .mlheir-card,
  .td-content .mlheir-figure,
  .td-content .mlheir-code,
  .td-content .mlheir-pipeline {
    background: var(--mlheir-card);
    border: 1px solid rgba(22, 24, 29, 0.08);
    border-radius: 1.15rem;
    box-shadow: var(--mlheir-shadow);
  }

  .td-content .mlheir-card,
  .td-content .mlheir-pipeline {
    padding: 1.15rem 1.2rem;
  }

  .td-content .mlheir-card h3,
  .td-content .mlheir-card h4,
  .td-content .mlheir-pipeline h3,
  .td-content .mlheir-pipeline h4 {
    margin-bottom: 0.6rem;
  }

  .td-content .mlheir-card ul,
  .td-content .mlheir-pipeline ul {
    margin: 0;
    padding-left: 1.2rem;
  }

  .td-content .mlheir-card li + li,
  .td-content .mlheir-pipeline li + li {
    margin-top: 0.32rem;
  }

  .td-content .mlheir-figure {
    overflow: hidden;
    margin: 0;
  }

  .td-content .mlheir-figure img {
    display: block;
    width: 100%;
    height: auto;
  }

  .td-content .mlheir-svg-figure {
    padding: 1rem;
    background: #f6f4ef;
  }

  .td-content .mlheir-svg-figure svg {
    display: block;
    width: 100%;
    height: auto;
  }

  .td-content .mlheir-svg-label {
    font-size: 15px;
    font-weight: 500;
  }

  .td-content .mlheir-svg-label-strong {
    font-size: 16px;
    font-weight: 600;
  }

  .td-content .mlheir-figure figcaption {
    padding: 0.8rem 1rem 0.95rem;
    color: var(--mlheir-muted);
    font-size: 0.92rem;
  }

  .td-content .mlheir-flow {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 0.85rem;
    align-items: start;
    margin-top: 0.8rem;
  }

  .td-content .mlheir-flow-step {
    position: relative;
    text-align: center;
  }

  .td-content .mlheir-flow-step:not(:last-child)::after {
    content: "";
    position: absolute;
    top: 1rem;
    right: -0.7rem;
    width: 1.2rem;
    border-top: 2px solid var(--mlheir-blue);
  }

  .td-content .mlheir-flow-chip {
    display: inline-block;
    width: 100%;
    padding: 0.7rem 0.5rem;
    border-radius: 0.8rem;
    background: var(--mlheir-blue);
    color: #fff;
    font-weight: 700;
  }

  .td-content .mlheir-flow-step p {
    margin: 0.55rem 0 0;
    font-size: 0.88rem;
  }

  .td-content .mlheir-transform-wrap {
    display: grid;
    grid-template-columns: minmax(0, 1fr) 120px;
    gap: 0.9rem;
    align-items: center;
  }

  .td-content .mlheir-transform-ladder {
    border: 2px dashed rgba(75, 162, 105, 0.5);
    border-radius: 1rem;
    padding: 0.85rem;
  }

  .td-content .mlheir-transform-step {
    text-align: center;
    padding: 0.7rem 0.75rem;
    border: 2px solid rgba(75, 162, 105, 0.75);
    border-radius: 0.7rem;
    color: #2b8d50;
    font-weight: 700;
    background: rgba(75, 162, 105, 0.04);
  }

  .td-content .mlheir-transform-step + .mlheir-transform-step {
    margin-top: 0.62rem;
  }

  .td-content .mlheir-transform-side {
    display: grid;
    gap: 1rem;
  }

  .td-content .mlheir-transform-side div {
    text-align: center;
    padding: 0.75rem 0.55rem;
    border-radius: 0.7rem;
    border: 2px solid rgba(228, 181, 76, 0.95);
    color: #de9802;
    background: rgba(228, 181, 76, 0.06);
    font-weight: 700;
  }

  .td-content .mlheir-code {
    overflow: auto;
    padding: 1rem 1.1rem;
    background: #f8fafc;
    margin: 0;
  }

  .td-content .mlheir-code pre {
    margin: 0;
    font-size: 0.88rem;
    line-height: 1.5;
    color: #38546a;
  }

  .td-content .mlheir-eq {
    margin: 1rem 0;
    padding: 1rem 1.1rem;
    border-radius: 0.95rem;
    border: 1px solid rgba(22, 24, 29, 0.07);
    background: #f6f4ef;
    overflow-x: auto;
  }

  .td-content .mlheir-eq-main,
  .td-content .mlheir-eq-line {
    font-family: "Times New Roman", Georgia, serif;
    color: #5e6470;
    line-height: 1.35;
  }

  .td-content .mlheir-eq-main {
    font-size: clamp(1.35rem, 2vw, 2rem);
  }

  .td-content .mlheir-eq-note {
    display: inline-block;
    margin: 0 0.9rem;
    font-weight: 600;
  }

  .td-content .mlheir-eq-system {
    margin-top: 0.7rem;
    padding-left: 1.1rem;
    border-left: 3px solid rgba(94, 100, 112, 0.18);
  }

  .td-content .mlheir-eq-line {
    font-size: clamp(1.1rem, 1.5vw, 1.45rem);
    white-space: nowrap;
  }

  .td-content .mlheir-eq-line + .mlheir-eq-line {
    margin-top: 0.35rem;
  }

  .td-content .mlheir-poster-eq {
    margin: 1rem auto 0.55rem;
    max-width: 43rem;
  }

  .td-content .mlheir-poster-eq img {
    width: 100%;
    height: auto;
    display: block;
  }

  .td-content .mlheir-layout-card h3 {
    font-size: clamp(2.2rem, 3vw, 3.1rem);
    line-height: 1.04;
    letter-spacing: -0.03em;
    margin-bottom: 0.9rem;
  }

  .td-content .mlheir-layout-card p,
  .td-content .mlheir-layout-card li {
    font-size: 1rem;
  }

  .td-content .mlheir-layout-card ul {
    margin-bottom: 0.8rem;
  }

  .td-content .mlheir-eq-plain {
    margin-top: 0.45rem;
    padding: 0;
    border: 0;
    border-radius: 0;
    background: transparent;
    box-shadow: none;
    overflow: visible;
  }

  .td-content .mlheir-eq-plain .mlheir-eq-line {
    font-size: clamp(1.55rem, 2vw, 2.2rem);
    text-align: center;
  }

  .td-content .mlheir-layer-stack {
    display: grid;
    gap: 0.7rem;
  }

  .td-content .mlheir-layer {
    padding: 0.8rem 0.9rem;
    border-radius: 0.9rem;
    border: 1px solid rgba(22, 24, 29, 0.08);
    background: linear-gradient(180deg, #fff 0%, #f7f7f7 100%);
  }

  .td-content .mlheir-layer strong {
    display: block;
    margin-bottom: 0.4rem;
    color: var(--mlheir-ink);
  }

  .td-content .mlheir-token-row {
    display: flex;
    flex-wrap: wrap;
    gap: 0.4rem;
  }

  .td-content .mlheir-token {
    display: inline-flex;
    align-items: center;
    padding: 0.24rem 0.55rem;
    border-radius: 999px;
    font-size: 0.84rem;
    line-height: 1.2;
    color: var(--mlheir-ink);
  }

  .td-content .mlheir-token-amber { background: rgba(228, 181, 76, 0.28); }
  .td-content .mlheir-token-gray { background: rgba(138, 145, 155, 0.2); }
  .td-content .mlheir-token-green { background: rgba(75, 162, 105, 0.2); }
  .td-content .mlheir-token-blue { background: rgba(74, 128, 236, 0.18); }
  .td-content .mlheir-token-violet { background: rgba(167, 120, 219, 0.16); }
  .td-content .mlheir-token-red { background: rgba(225, 91, 72, 0.18); }

  .td-content .mlheir-logo-card {
    display: grid;
    gap: 0.9rem;
    align-content: start;
  }

  .td-content .mlheir-flat {
    background: transparent;
    border: 0;
    box-shadow: none;
    padding: 0;
  }

  .td-content .mlheir-logo-card img {
    max-width: 100%;
    height: auto;
    display: block;
  }

  .td-content .mlheir-logo-dark {
    background: #101318;
    color: #e7edf7;
  }

  .td-content .mlheir-logo-dark p,
  .td-content .mlheir-logo-dark li {
    color: #c8d3e3;
  }

  .td-content .mlheir-links {
    display: flex;
    flex-wrap: wrap;
    gap: 0.7rem;
    margin-top: 1rem;
  }

  .td-content .mlheir-linkbox {
    flex: 1 1 150px;
    min-width: 140px;
    padding: 0.9rem 1rem;
    border-radius: 0.95rem;
    border: 1px solid rgba(22, 24, 29, 0.08);
    background: #fff;
    box-shadow: var(--mlheir-shadow);
    text-decoration: none;
  }

  .td-content .mlheir-linkbox strong {
    display: block;
    color: var(--mlheir-ink);
  }

  .td-content .mlheir-linkbox span {
    color: var(--mlheir-muted);
    font-size: 0.92rem;
  }

  .td-content .mlheir-project-grid {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 0.9rem;
    margin: 1rem 0 1.5rem;
  }

  .td-content .mlheir-project {
    padding: 0;
    border-radius: 0;
    background: transparent;
    border: 0;
    box-shadow: none;
  }

  .td-content .mlheir-project h4 {
    margin-bottom: 0.45rem;
    color: var(--mlheir-blue);
    font-size: 1rem;
  }

  .td-content .mlheir-project p {
    margin: 0;
    font-size: 0.92rem;
  }

  .td-content .mlheir-stack-split {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 0.7rem;
  }

  .td-content .mlheir-logo-note {
    font-size: 0.94rem;
  }

  .td-content .mlheir-logo-note p:last-child {
    margin-bottom: 0;
  }

  @media (max-width: 1450px) {
    .td-content .mlheir-grid-four {
      grid-template-columns: repeat(3, minmax(0, 1fr));
    }
  }

  @media (max-width: 1100px) {
    .td-content .mlheir-hero,
    .td-content .mlheir-grid-two,
    .td-content .mlheir-grid-three,
    .td-content .mlheir-grid-four,
    .td-content .mlheir-grid-hardware,
    .td-content .mlheir-project-grid,
    .td-content .mlheir-transform-wrap {
      grid-template-columns: 1fr;
    }

    .td-content .mlheir-hero-side {
      justify-items: start;
    }

    .td-content .mlheir-flow {
      grid-template-columns: 1fr;
    }

    .td-content .mlheir-flow-step:not(:last-child)::after {
      display: none;
    }

    .td-content .mlheir-stack-split {
      grid-template-columns: 1fr;
    }
  }
</style>

<div class="mlheir">
  <div class="mlheir-hero">
    <div>
      <h1>HEIR: Fully Homomorphic Machine Learning with a Universal Compiler</h1>
      <p class="mlheir-lede">
        An FHE compiler toolchain and development platform without sacrificing
        generality and extensibility.
      </p>
      <p class="mlheir-meta">
        <span>HEIR project</span>
        <a href="mailto:asraa@google.com">asraa@google.com</a>
        <a href="https://arxiv.org/abs/2508.11095">arXiv: 2508.11095</a>
      </p>
    </div>
    <div class="mlheir-hero-side">
      <img src="/images/heir_logo.svg" alt="HEIR logo" />
      <div class="mlheir-side-note">
        <p>
          HEIR provides an MLIR-based path from ML frontends to scheme-level
          IRs, library backends, and lower-level arithmetic intended for
          hardware integration.
        </p>
      </div>
    </div>
  </div>

<div class="mlheir-band">
    <h2 id="ml-frontend" class="mlheir-band-heading">
      ML Frontend
      <a class="mlheir-band-anchor" href="#ml-frontend" aria-label="Link to ML Frontend">#</a>
    </h2>
  </div>
</div>

Torch models are converted with `torch-mlir` to `linalg` on tensors. HEIR also
accepts inputs that pass through `StableHLO` and `onnx-mlir`; the `tensor` and
`arith` dialects are part of the entry-level IR used by the pipeline.

<div class="mlheir">
  <div class="mlheir-grid mlheir-grid-hardware">
    <figure class="mlheir-figure mlheir-svg-figure">
      <img src="/images/ml_with_heir/frontend_flow.svg" alt="ML frontend flow from PyTorch, TensorFlow, and ONNX into HEIR's linalg entry level" />
      <figcaption>
        PyTorch, TensorFlow, and ONNX converge into HEIR's linalg entry level
        through torch-mlir, onnx-mlir, and StableHLO.
      </figcaption>
    </figure>
    <div class="mlheir-grid">
      <div class="mlheir-card">
        <h3>Linalg Entry Level</h3>
        <p>
          The <code>linalg</code> dialect is a funnel dialect for HEIR's ML
          frontend. Its abstraction level is useful for matching on ML kernel
          operations during optimization. Canonicalization patterns at this
          level simplify memory shuffling and reduce non-linear operations
          before scheme-specific lowering.
        </p>
      </div>
      <div class="mlheir-card">
        <h3>Compilation Configuration</h3>
        <ul>
          <li>Backend and scheme selection</li>
          <li>Secret input data selected with annotations</li>
          <li>Non-linear activation approximation degree</li>
          <li>Range bounds from model metadata</li>
          <li>User controlled kernel selection</li>
        </ul>
      </div>
    </div>
  </div>
</div>

### Annotated MLIR Sketch

The following annotated MLIR sketch shows where backend, scheme, secret inputs,
kernel choices, and activation bounds enter the pipeline. It is schematic rather
than parser-ready MLIR.

```mlir
module attributes {backend.openfhe, scheme.ckks} {
  func.func @mnist(%input: tensor<784> {secret.secret}) -> tensor<10> {
    %matrix1 = arith.constant dense<...> : tensor<512x784>
    %bias1 = arith.constant dense<...> : tensor<512>
    %matrix2 = arith.constant dense<...> : tensor<10x512>
    %bias2 = arith.constant dense<...> : tensor<10>
    %cst = arith.constant dense<0.0> : tensor<512>
    %0 = linalg.matvec ins(%matrix1, %input) {kernel = "diagonal"}
    %1 = arith.addf %0, %bias1
    %2 = arith.maximumf %1, %cst {degree = 6, lower = -15.0, upper = 12}
    %3 = linalg.matvec ins(%matrix2, %2)
    %4 = arith.addf %3, %bias2
    return %4
  }
}
```

<div class="mlheir">
  <div class="mlheir-band mlheir-band-gold">
    <h2 id="ml-compilation-pipeline" class="mlheir-band-heading">
      ML Compilation Pipeline
      <a class="mlheir-band-anchor" href="#ml-compilation-pipeline" aria-label="Link to ML Compilation Pipeline">#</a>
    </h2>
  </div>
</div>

<div class="mlheir">
<div class="mlheir-grid mlheir-layout-row">
<div class="mlheir-card mlheir-layout-card">
<h3>Relation-based Ciphertext Layouts</h3>
<p>
Layouts are a <em>partial function</em> mapping from the index set of a
cleartext tensor to the index set of a list of ciphertext slots using
<em>Presburger relations</em> and quasi-affine formulas.
</p>
<ul>
<li>Fully general layout annotations describing plaintext-ciphertext relation</li>
<li>Polyhedral optimization with Integer Set Library analyzes and manipulates layouts, for e.g. to compute kernel simplifications or slot utilization for batching</li>
</ul>
<p>One useful example maps an <code>(i, j)</code> index in an <code>8 x 8</code> tensor to eight ciphertexts with 1024 slots:</p>
<figure class="mlheir-poster-eq">
<img src="/images/ml_with_heir/relation_layout.svg" alt="Mapping from tensor indices i and j to ciphertext and slot with modular constraints" />
</figure>
<p>Related indexing formulas, adapted from Orion, are:</p>
<div class="mlheir-eq mlheir-eq-plain">
<div class="mlheir-eq-line">m<sub>r</sub> = (i<sub>dr</sub> + P)F + i<sub>dc</sub> + P</div>
<div class="mlheir-eq-line">m<sub>c</sub> = W<sub>d</sub>i<sub>dr</sub> + i<sub>dc</sub> + W<sub>d</sub>i<sub>fr</sub> + i<sub>fc</sub></div>
</div>
</div>
<div class="mlheir-card">
<h3>New Layout Integrations</h3>
<p>
HEIR integrates
<a href="https://doi.org/10.1109/TIFS.2024.3490862">bicyclic</a> [1]
and
<a href="https://eprint.iacr.org/2025/1200">tricyclic</a> [2]
layouts and kernels to compute batched matrix multiplication for
parallelized multi-head self-attention with optimal multiplicative
depth.
</p>
<p>
Supported layouts and kernels are easily extended with ISL utilities
and a testable MLIR-agnostic kernel library.
</p>
</div>
</div>
</div>

<div class="mlheir">
  <div class="mlheir-pipeline">
    <h3>Layout Optimization Flow</h3>
    <div class="mlheir-flow">
      <div class="mlheir-flow-step">
        <div class="mlheir-flow-chip">Propagate</div>
        <p>Forward analysis propagates IR with default layouts and kernels.</p>
      </div>
      <div class="mlheir-flow-step">
        <div class="mlheir-flow-chip">Optimize</div>
        <p>Cost models select optimal kernels to minimize cost and layout conversions.</p>
      </div>
      <div class="mlheir-flow-step">
        <div class="mlheir-flow-chip">Simplify</div>
        <p>Backwards traversal hoists layout conversions to encodings.</p>
      </div>
    </div>
  </div>
</div>

<div class="mlheir">
  <div class="mlheir-grid mlheir-grid-two">
    <div class="mlheir-card">
      <h3>Optimization Variety Pack</h3>
      <p>HEIR's ML pipeline uses a number of generally applicable optimization patterns:</p>
      <ul>
        <li>Sparse matrix product simplification</li>
        <li>Baby-step giant-step for general reductions</li>
        <li>Minimal depth polynomials evaluation with Paterson-Stockmeyer</li>
        <li>Fast (hoisted) rotation rewrites</li>
        <li>Minimized extended key basis switching</li>
        <li>High level program vectorization</li>
        <li>Shift networks for layout conversions</li>
        <li>Loop support with HALO optimizations</li>
        <li>Multiplexed data packing for slot utilization</li>
      </ul>
    </div>
    <div class="mlheir-pipeline">
      <h3>Transform Ladder</h3>
      <div class="mlheir-transform-wrap">
        <div class="mlheir-transform-ladder">
          <div class="mlheir-transform-step">Model Transforms</div>
          <div class="mlheir-transform-step">Arithmetization</div>
          <div class="mlheir-transform-step">Vectorization</div>
          <div class="mlheir-transform-step">Layout Pipeline</div>
          <div class="mlheir-transform-step">Noise Management</div>
          <div class="mlheir-transform-step">Parameter Selection</div>
        </div>
        <div class="mlheir-transform-side">
          <div>Plaintext Execution</div>
          <div>Scheme IR</div>
        </div>
      </div>
    </div>
  </div>

<div class="mlheir-grid mlheir-grid-two">
    <div class="mlheir-card">
      <h3>Make It Easy</h3>
      <p>HEIR simplifies the developer and debugging experience with:</p>
      <ul>
        <li>Tracking and debugging utilities from MLIR</li>
        <li>Plaintext execution mode with custom debug handlers</li>
        <li>Client helpers for encoding and encryption/decryption</li>
        <li>Output code that remains human-readable enough to inspect and modify</li>
        <li>Cleartext computations hoisted into separate functions for precomputation</li>
        <li>Scheme-specific parameter selection</li>
      </ul>
    </div>
    <div class="mlheir-code">
      <pre><code>// preprocessing functions
Plaintext matvec__preprocessing(CryptoContext cc) {
  const auto&amp; pt2 = cc-&gt;MakeCKKSPackedPlaintext(c0);
  return pt2;
}

// main workload Ciphertext matvec(CryptoContext cc, Ciphertext ct) { const
auto& ct5 = cc->EvalMult(ct4, pt2); const auto& ct6 = cc->EvalRotate(ct, 3);
const auto& ct47 = cc->EvalAdd(ct38, ct46); const auto& ct48 =
cc->EvalMultNoRelin(ct47, ct47); const auto& ct49 = cc->Relinearize(ct48); ... }

// client functions Ciphertext matvec\_\_encrypt\_\_arg0( CryptoContext cc,
std::vector\<float> v0, PublicKey pk); std::vector\<float>
matvec\_\_decrypt\_\_result0( CryptoContext cc, Ciphertext ct, PrivateKey sk);
CryptoContext matvec\_\_generate_crypto_context(); CryptoContext
matvec\_\_configure_crypto_context();</code></pre> </div>

</div>
</div>

<div class="mlheir">
  <div class="mlheir-band mlheir-band-red">
    <h2 id="hardware-integrations" class="mlheir-band-heading">
      Hardware Integrations
      <a class="mlheir-band-anchor" href="#hardware-integrations" aria-label="Link to Hardware Integrations">#</a>
    </h2>
  </div>
</div>

<div class="mlheir">
  <div class="mlheir-grid mlheir-grid-two">
    <div class="mlheir-card mlheir-flat">
      <div class="mlheir-layer-stack">
        <div class="mlheir-layer">
          <strong>ML frontends</strong>
          <div class="mlheir-token-row">
            <span class="mlheir-token mlheir-token-amber">Python</span>
            <span class="mlheir-token mlheir-token-amber">Torch</span>
            <span class="mlheir-token mlheir-token-amber">TensorFlow Lite</span>
          </div>
        </div>
        <div class="mlheir-layer">
          <strong>Standard MLIR</strong>
          <div class="mlheir-token-row">
            <span class="mlheir-token mlheir-token-gray">func</span>
            <span class="mlheir-token mlheir-token-gray">linalg</span>
            <span class="mlheir-token mlheir-token-gray">tensor</span>
            <span class="mlheir-token mlheir-token-gray">arith</span>
            <span class="mlheir-token mlheir-token-gray">affine</span>
          </div>
        </div>
        <div class="mlheir-layer">
          <strong>Secret arithmetic</strong>
          <div class="mlheir-token-row">
            <span class="mlheir-token mlheir-token-green">secret</span>
            <span class="mlheir-token mlheir-token-green">tensor_ext</span>
            <span class="mlheir-token mlheir-token-green">mgmt</span>
            <span class="mlheir-token mlheir-token-green">polynomial</span>
            <span class="mlheir-token mlheir-token-green">comb</span>
          </div>
        </div>
        <div class="mlheir-layer">
          <strong>Scheme APIs</strong>
          <div class="mlheir-token-row">
            <span class="mlheir-token mlheir-token-blue">lwe</span>
            <span class="mlheir-token mlheir-token-blue">bgv</span>
            <span class="mlheir-token mlheir-token-blue">ckks</span>
            <span class="mlheir-token mlheir-token-blue">cggi</span>
          </div>
        </div>
        <div class="mlheir-stack-split">
          <div class="mlheir-layer">
            <strong>Scheme implementation</strong>
            <div class="mlheir-token-row">
              <span class="mlheir-token mlheir-token-violet">polynomial</span>
              <span class="mlheir-token mlheir-token-violet">rns</span>
              <span class="mlheir-token mlheir-token-violet">mod_arith</span>
            </div>
          </div>
          <div class="mlheir-layer">
            <strong>Library APIs</strong>
            <div class="mlheir-token-row">
              <span class="mlheir-token mlheir-token-red">lattigo</span>
              <span class="mlheir-token mlheir-token-red">tfhe_rust</span>
              <span class="mlheir-token mlheir-token-red">jaxite</span>
              <span class="mlheir-token mlheir-token-red">openfhe</span>
            </div>
          </div>
        </div>
        <div class="mlheir-layer">
          <strong>Hardware dialects</strong>
          <div class="mlheir-token-row">
            <span class="mlheir-token mlheir-token-red">llvm</span>
            <span class="mlheir-token mlheir-token-red">scifr</span>
            <span class="mlheir-token mlheir-token-red">...</span>
          </div>
        </div>
      </div>
    </div>
    <div class="mlheir-grid">
      <div class="mlheir-card">
        <h3>Exit Dialects</h3>
        <p>
          Support for multiple backends (CPU, GPU, FPGA, ASICs, and photonics)
          allows for comprehensive testing and benchmarking. After HEIR's high
          level program analysis and compilation, data layouts, kernels,
          schemes, and parameters are selected and the IR uses scheme level
          operations. There are two possible ways to exit HEIR:
        </p>
        <ol>
          <li>
            <strong>Library dialects</strong> (e.g. Lattigo, OpenFHE,
            <code>tfhe-rs</code>) mirror APIs and are translated to code via
            HEIR's emitter. Allows fast prototyping and easy integration but
            limits the ability to perform fusion or other cross-operation
            optimizations.
          </li>
          <li>
            <strong>Low level IRs:</strong> scheme operations are implemented
            using polynomial and modular arithmetic dialects. Hardware specific
            toolchains handle further optimization, scheduling and assembly
            (e.g. the LLVM toolchain compiles the MLIR for CPU). This path is
            suitable for longer term, robust integrations.
          </li>
        </ol>
      </div>
      <div class="mlheir-grid mlheir-grid-four">
        <div class="mlheir-card mlheir-logo-card">
          <h4>Optalysys</h4>
          <img src="/images/ml_with_heir/optalysys.jpg" alt="Optalysys logo" />
          <p>
            Optalysys utilizes <em>photonic computing technology</em> to
            perform modular arithmetic operations over the Polynomial Modular
            Number System (PMNS). Integration with HEIR's generated low level
            NTT and mod arith code will allow running FHE workloads on
            Optalysys' optical processing chips.
          </p>
        </div>
        <div class="mlheir-card mlheir-logo-card">
          <h4>Belfort</h4>
          <p>
            Belfort's integrates their FPGA-based accelerator with HEIR
            through the CGGI boolean and shortint APIs. They utilize
            vectorization strategies in HEIR and software optimizations in
            their custom <code>tfhe-rs</code> library for performance.
          </p>
        </div>
        <div class="mlheir-card mlheir-logo-card mlheir-logo-note">
          <p>
            Plus more backends in progress (e.g. FIDESlib [1] GPU backend) and
            under NDA.
          </p>
          <p>
            C. Aguilo-Domingo et al., "FIDESlib: A Fully-Fledged Open-Source
            FHE Library for Efficient CKKS on GPUs," in 2025 IEEE
            International Symposium on Performance Analysis of Systems and
            Software (ISPASS), Ghent, Belgium, 2025, pp. 1-3.
          </p>
        </div>
        <div class="mlheir-card mlheir-logo-card">
          <h4>CROSS</h4>
          <p>
            TPU-native CKKS implementation with SoTA performance vs GPU (20ms
            bootstrap) using JAX. HEIR integrations utilizes the CKKS dialect
            to lower to the CROSS API exit dialect.
          </p>
        </div>
        <div class="mlheir-card mlheir-logo-card">
          <h4>Cornami</h4>
          <img src="/images/ml_with_heir/cornami.jpg" alt="Cornami logo" />
          <p>
            Cornami's MX2 systolic array is integrated as a backend to HEIR's
            MLIR pipeline for CGGI and CKKS schemes. HEIR exits to Cornami's
            Secure Computing Interface Framework (SCIFR) with custom
            optimizations.
          </p>
        </div>
        <div class="mlheir-card mlheir-logo-card">
          <img src="/images/ml_with_heir/fhetch.png" alt="FHE Tech logo" />
          <p>The global FHE hardware consortium (www.fhetch.org)</p>
          <p>
            HEIR tracks progress of the <em>polynomial intermediate
            representation</em> (IR) developed by FHE Technical Consortium for
            Hardware (FHETCH). The IR aims to provide an standardised set of
            hardware-level operations for interoperable platform integration.
            HEIR's <em>polynomial</em> dialect aligns with the evolving
            standard.
          </p>
        </div>
      </div>
    </div>
  </div>
</div>

<div class="mlheir">
  <div class="mlheir-band mlheir-band-green">
    <h2 id="community" class="mlheir-band-heading">
      Community
      <a class="mlheir-band-anchor" href="#community" aria-label="Link to Community">#</a>
    </h2>
  </div>
</div>

<div class="mlheir">
  <div class="mlheir-grid mlheir-grid-two">
    <div class="mlheir-card mlheir-flat">
      <p>
        HEIR's open-source framework supports major homomorphic encryption
        methods, enabling efficient research and benchmarking. Its architecture
        facilitates the integration of state of the art and emerging
        methodologies, as evidenced by various projects built with or
        incorporated into HEIR.
      </p>
      <div class="mlheir-links">
        <a class="mlheir-linkbox" href="https://github.com/google/heir">
          <strong>GitHub</strong>
          <span>github.com/google/heir</span>
        </a>
        <a class="mlheir-linkbox" href="https://discord.com/invite/fhe-org">
          <strong>Discord</strong>
          <span>discord.com/invite/fhe-org</span>
        </a>
        <a class="mlheir-linkbox" href="/community/">
          <strong>Community</strong>
          <span>google.github.io/heir/community</span>
        </a>
      </div>
    </div>
    <div class="mlheir-card mlheir-flat">
      <h3>Call for Contributions</h3>
      <p>
        Connect with us to explore potential research directions and
        integrations, including:
      </p>
      <ul>
        <li>Integrating Gentry-Lee FHE scheme "Fully Homomorphic Encryption for Matrix Arithmetic"</li>
        <li>Layout optimizer that uses the structure of Presburger relations, and/or the general joint layout+kernel selection problem</li>
        <li>New FHE scheme implementations (e.g. GBFV) and optimizations</li>
        <li>Incorporating memory constraints into cost models for kernel compilation</li>
        <li>Profile-guided optimizations for parameter selection &amp; scale management</li>
      </ul>
    </div>
  </div>

<div class="mlheir-project-grid">
    <div class="mlheir-project">
      <h4>Fhelipe Layout Hoisting</h4>
      <p>HEIR uses a FHelipe's hoisting heuristic to minimize layout conversions between operations.</p>
    </div>
    <div class="mlheir-project">
      <h4>Average-Case Noise Analysis</h4>
      <p>HEIR was used to experimentally demonstrate underestimations of average-case noise analysis.</p>
    </div>
    <div class="mlheir-project">
      <h4>HALO Compiler Loop Support</h4>
      <p>HEIR adopts transforms from the HALO compiler for loop-aware bootstrapping placement.</p>
    </div>
    <div class="mlheir-project">
      <h4>ROTOM: Autovectorizing HE</h4>
      <p>ROTOM's tensor vectorization strategy is integrated as an option for layout optimization.</p>
    </div>
    <div class="mlheir-project">
      <h4>Orion Compiler Kernels</h4>
      <p>HEIR incorporates Orion's convolution data layout and kernel with double-hoisting and BSGS.</p>
    </div>
    <div class="mlheir-project">
      <h4>KeyMemRT Memory Scalability</h4>
      <p>Key memory management minimization strategies are incorporated into HEIR.</p>
    </div>
    <div class="mlheir-project">
      <h4>Tricycle: Private Transformers</h4>
      <p>HEIR supports tricyclic layouts to enable ciphertext matrix multiplications for self-attention.</p>
    </div>
    <div class="mlheir-project">
      <h4>Vos-Vos-Erkin Shift Networks</h4>
      <p>Efficient shift network implementation of layout conversions using graph coloring.</p>
    </div>
  </div>
</div>

## References

1. J. Chen, L. Yang, W. Wu, Y. Liu, and Y. Feng,
   [Homomorphic Matrix Operations Under Bicyclic Encoding](https://doi.org/10.1109/TIFS.2024.3490862),
   *IEEE Transactions on Information Forensics and Security*, 2025.
1. L. Lim et al.,
   [Tricycle: Private Transformer Inference with Tricyclic Encodings](https://eprint.iacr.org/2025/1200),
   *IACR ePrint*, 2025.

<!-- mdformat global-off -->
