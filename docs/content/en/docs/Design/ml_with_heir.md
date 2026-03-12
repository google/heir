---
title: ML with HEIR
linkTitle: ML with HEIR
weight: 12
description: >-
  HEIR's ML frontend, compilation pipeline, hardware integrations, and active
  research directions.
---

<!-- mdformat off -->

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

  body:has(.mlheir-hero) .td-main main,
  body.mlheir-page-body .td-main main {
    flex: 1 1 0% !important;
    max-width: 100% !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
  }

  @media (max-width: 768px) {
    body:has(.mlheir-hero) .td-main main,
    body.mlheir-page-body .td-main main {
      padding-left: 1rem !important;
      padding-right: 1rem !important;
    }
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
    justify-items: center;
  }

  .td-content .mlheir-hero-side img {
    width: min(100%, 240px);
    height: auto;
  }

  .td-content .mlheir-side-note {
    width: min(100%, 21rem);
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
  }

  .td-content .mlheir h2.mlheir-band-heading {
    display: flex;
    align-items: center;
    gap: 0.55rem;
    width: 100%;
    padding: 0.52rem 1.15rem;
    border-radius: 999px;
    background: var(--mlheir-blue);
    color: #fff;
    font-size: 1rem;
    font-weight: 700;
    letter-spacing: 0.01em;
    margin: 0;
    scroll-margin-top: 5rem;
  }

  .td-content .mlheir-band-gold h2.mlheir-band-heading {
    background: var(--mlheir-gold);
  }

  .td-content .mlheir-band-red h2.mlheir-band-heading {
    background: var(--mlheir-red);
  }

  .td-content .mlheir-band-green h2.mlheir-band-heading {
    background: var(--mlheir-green);
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
    align-items: stretch;
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
    display: flex;
    flex-direction: column;
  }

  .td-content .mlheir-grid-hardware > .mlheir-figure {
    height: 100%;
    margin: 0;
  }

  .td-content .mlheir-grid-hardware > .mlheir-figure img {
    flex: 1;
    object-fit: contain;
  }

  .td-content .mlheir-grid-hardware > .mlheir-grid {
    grid-template-rows: 1fr 1fr;
    margin: 0;
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

  /* Outer grid: [dashed-ladder] [side-boxes] */
  .td-content .mlheir-transform-layout {
    display: grid;
    grid-template-columns: minmax(0, 1fr) 130px;
    grid-template-rows: 1fr auto;
    column-gap: 0.75rem;
  }

  /* Dashed border wraps ONLY the ladder column */
  .td-content .mlheir-transform-inner {
    grid-column: 1;
    grid-row: 1;
    border: 2px dashed rgba(75, 162, 105, 0.5);
    border-radius: 1rem;
    padding: 0.85rem;
  }

  .td-content .mlheir-transform-ladder {
    display: flex;
    flex-direction: column;
    align-items: stretch;
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

  .td-content .mlheir-transform-arrow {
    text-align: center;
    color: rgba(75, 162, 105, 0.8);
    font-size: 0.9rem;
    line-height: 1;
    margin: 0.05rem 0;
  }

  /* Side boxes column — outside the dashed border */
  .td-content .mlheir-transform-side {
    grid-column: 2;
    grid-row: 1;
    display: flex;
    flex-direction: column;
    align-self: stretch;
  }

  .td-content .mlheir-transform-side-spacer {
    min-height: 0;
  }

  .td-content .mlheir-transform-side-item {
    display: flex;
    align-items: center;
    margin-left: -0.75rem; /* pull back into column gap to draw line */
  }

  .td-content .mlheir-transform-side-item::before {
    content: "";
    flex-shrink: 0;
    width: 0.75rem; /* matches column-gap */
    border-top: 2px dashed rgba(228, 181, 76, 0.85);
  }

  .td-content .mlheir-transform-side-item > div {
    flex: 1;
    text-align: center;
    padding: 0.75rem 0.55rem;
    border-radius: 0.7rem;
    border: 2px solid rgba(228, 181, 76, 0.95);
    color: #de9802;
    background: rgba(228, 181, 76, 0.06);
    font-weight: 700;
  }

  .td-content .mlheir-transform-caption {
    grid-column: 1;
    grid-row: 2;
    text-align: center;
    font-size: 0.8rem;
    color: var(--mlheir-muted);
    margin-top: 0.75rem;
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

  /* Style Hugo-generated highlight blocks between mlheir sections */
  .td-content .mlheir + .highlight,
  .td-content .mlheir + p + .highlight {
    background: var(--mlheir-card, #ffffff);
    border: 1px solid rgba(22, 24, 29, 0.08);
    border-radius: 1.15rem;
    box-shadow: 0 16px 36px rgba(22, 24, 29, 0.08);
    overflow: auto;
    padding: 1rem 1.1rem;
    margin-bottom: 1.75rem;
  }

  .td-content .mlheir + .highlight pre,
  .td-content .mlheir + p + .highlight pre {
    margin: 0;
    font-size: 0.88rem;
    line-height: 1.5;
    background: transparent !important;
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

  .td-content .mlheir-setbuilder {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0;
    font-family: "STIX Two Text", "Times New Roman", Georgia, serif;
    color: var(--mlheir-ink);
    margin: 1rem 0 0.55rem;
    overflow-x: auto;
  }

  .td-content .mlheir-setbuilder-brace {
    width: 14px;
    flex-shrink: 0;
    align-self: stretch;
    color: #888;
    margin: 0.4rem 0.15rem;
  }

  .td-content .mlheir-setbuilder-map {
    font-size: clamp(1.1rem, 1.6vw, 1.5rem);
    padding: 0 0.5rem;
    white-space: nowrap;
    align-self: center;
  }

  .td-content .mlheir-setbuilder-divider {
    width: 1.5px;
    align-self: stretch;
    background: #44484f;
    margin: 0.6rem 0;
    flex-shrink: 0;
  }

  .td-content .mlheir-setbuilder-conditions {
    display: flex;
    flex-direction: column;
    gap: 0.2rem;
    padding: 0.3rem 0.5rem;
  }

  .td-content .mlheir-setbuilder-conditions span {
    font-size: clamp(1.05rem, 1.5vw, 1.4rem);
    white-space: nowrap;
    line-height: 1.45;
  }

  .td-content .mlheir-setbuilder-caption {
    text-align: center;
    font-size: 0.92rem;
    color: var(--mlheir-muted);
    margin-top: 0.3rem;
  }

  .td-content .mlheir-layout-card h3 {
    font-size: clamp(1.4rem, 2vw, 1.75rem);
    line-height: 1.15;
    letter-spacing: -0.02em;
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

  .td-content .mlheir-layer-stack-wrap {
    display: flex;
    gap: 0.5rem;
  }

  .td-content .mlheir-layer-stack-arrow {
    flex-shrink: 0;
    width: 1.8rem;
    display: flex;
    align-items: stretch;
    justify-content: center;
    position: relative;
  }

  .td-content .mlheir-layer-stack-arrow::before {
    content: "";
    display: block;
    width: 3px;
    background: linear-gradient(180deg, #b0b8c4 0%, #6b7a8d 100%);
    border-radius: 2px;
  }

  .td-content .mlheir-layer-stack-arrow::after {
    content: "";
    position: absolute;
    bottom: 0;
    left: 50%;
    transform: translateX(-50%);
    width: 0;
    height: 0;
    border-left: 7px solid transparent;
    border-right: 7px solid transparent;
    border-top: 10px solid #6b7a8d;
  }

  .td-content .mlheir-layer-stack {
    display: grid;
    gap: 0.7rem;
    flex: 1;
    min-width: 0;
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
    text-align: center;
  }

  .td-content .mlheir-token-row {
    display: flex;
    flex-wrap: wrap;
    gap: 0.4rem;
    justify-content: center;
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

  .td-content .mlheir-token-standalone {
    border-radius: 0.5rem;
    border: 1px solid rgba(22, 24, 29, 0.12);
    padding: 0.35rem 0.7rem;
    font-size: 0.88rem;
  }

  .td-content .mlheir-token-amber { background: rgba(228, 181, 76, 0.28); }
  .td-content .mlheir-token-gray { background: rgba(138, 145, 155, 0.2); }
  .td-content .mlheir-token-green { background: rgba(75, 162, 105, 0.2); }
  .td-content .mlheir-token-blue { background: rgba(74, 128, 236, 0.18); }
  .td-content .mlheir-token-violet { background: rgba(167, 120, 219, 0.16); }
  .td-content .mlheir-token-red { background: rgba(225, 91, 72, 0.18); }

  .td-content .mlheir-layer-amber { background: rgba(228, 181, 76, 0.15); border-color: rgba(228, 181, 76, 0.3); }
  .td-content .mlheir-layer-gray { background: rgba(138, 145, 155, 0.1); border-color: rgba(138, 145, 155, 0.2); }
  .td-content .mlheir-layer-green { background: rgba(75, 162, 105, 0.1); border-color: rgba(75, 162, 105, 0.2); }
  .td-content .mlheir-layer-blue { background: rgba(74, 128, 236, 0.1); border-color: rgba(74, 128, 236, 0.2); }
  .td-content .mlheir-layer-violet { background: rgba(167, 120, 219, 0.1); border-color: rgba(167, 120, 219, 0.2); }
  .td-content .mlheir-layer-red { background: rgba(225, 91, 72, 0.1); border-color: rgba(225, 91, 72, 0.2); }

  .td-content .mlheir-logo-card {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 0.9rem;
  }

  .td-content .mlheir-logo-card > img,
  .td-content .mlheir-logo-card > .mlheir-logo-on-dark > img {
    flex-shrink: 0;
    width: auto;
    max-width: 100%;
    height: 56px;
    object-fit: contain;
    object-position: center;
    margin: 0 auto;
    display: block;
  }

  .td-content .mlheir-logo-card > .mlheir-wordmark {
    flex-shrink: 0;
    height: 56px;
    display: flex;
    align-items: center;
    justify-content: center;
  }

  /* Fixed-height logo zone so centerlines align across cards */
  .td-content .mlheir-logo-card > img,
  .td-content .mlheir-logo-card > .mlheir-wordmark,
  .td-content .mlheir-logo-card > .mlheir-logo-on-dark {
    height: 56px;
    margin-top: 0.5rem;
    margin-bottom: 0.5rem;
  }

  .td-content .mlheir-logo-card > p:last-child {
    margin-top: auto;
    margin-bottom: 0;
  }

  .td-content .mlheir-flat {
    background: transparent;
    border: 0;
    box-shadow: none;
    padding: 0;
  }

  .td-content .mlheir-logo-card p img {
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
    overflow: hidden;
  }

  .td-content .mlheir-linkbox strong {
    display: block;
    color: var(--mlheir-ink);
  }

  .td-content .mlheir-linkbox span {
    color: var(--mlheir-muted);
    font-size: 0.85rem;
    display: block;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .td-content .mlheir-linkicon {
    width: 28px;
    height: 28px;
    color: var(--mlheir-muted);
    margin-bottom: 0.3rem;
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

  .td-content .mlheir-stack-split-left {
    display: grid;
    gap: 0.7rem;
    align-content: start;
  }

  .td-content .mlheir-logo-note {
    font-size: 0.94rem;
  }

  .td-content .mlheir-logo-note p:last-child {
    margin-bottom: 0;
  }

  .td-content .mlheir-card-two-col {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1.5rem;
    align-items: start;
  }

  .td-content .mlheir-card-sidebar {
    display: grid;
    grid-template-columns: 1fr 2fr;
    gap: 1.5rem;
    align-items: start;
  }

  /* Prevent grid children from overflowing (pre/code blocks) */
  .td-content .mlheir-card-two-col > *,
  .td-content .mlheir-card-sidebar > *,
  .td-content .mlheir-card-main-sidebar > *,
  .td-content .mlheir-card-auto-content > * {
    min-width: 0;
  }

  .td-content .mlheir-card .highlight,
  .td-content .mlheir-card .highlight pre {
    overflow-x: auto;
  }

  .td-content .mlheir-card-main-sidebar {
    display: grid;
    grid-template-columns: minmax(0, 1.4fr) minmax(280px, 1fr);
    gap: 1.25rem;
    align-items: start;
  }

  .td-content .mlheir-card-auto-content {
    display: grid;
    grid-template-columns: auto 1fr;
    gap: 1.25rem;
    align-items: start;
  }

  .td-content .mlheir-wordmark {
    font-size: 1.6rem;
    font-weight: 800;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
    text-align: center;
  }

  .td-content .mlheir-logo-on-dark {
    background: #101318;
    border-radius: 0.7rem;
    padding: 0.5rem 0.7rem;
  }

  @media (max-width: 1450px) {
    .td-content .mlheir-grid-four {
      grid-template-columns: repeat(3, minmax(0, 1fr));
    }
  }

  @media (max-width: 1100px) {
    .td-content .mlheir-vendor-grid {
      grid-template-columns: repeat(2, minmax(0, 1fr));
    }
  }

  @media (max-width: 600px) {
    .td-content .mlheir-vendor-grid {
      grid-template-columns: 1fr;
    }
  }

  @media (max-width: 1100px) {
    .td-content .mlheir-hero,
    .td-content .mlheir-grid-two,
    .td-content .mlheir-grid-three,
    .td-content .mlheir-grid-four,
    .td-content .mlheir-grid-hardware,
    .td-content .mlheir-project-grid,
    .td-content .mlheir-transform-layout,
    .td-content .mlheir-card-two-col,
    .td-content .mlheir-card-sidebar,
    .td-content .mlheir-card-main-sidebar,
    .td-content .mlheir-card-auto-content {
      grid-template-columns: 1fr;
    }

    .td-content .mlheir-hero-side {
      align-items: stretch;
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

<article>
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
    </h2>
  </div>
  <div class="mlheir-card mlheir-card-two-col" style="margin: 1rem 0 1.5rem;">
    <figure class="mlheir-figure mlheir-svg-figure" style="margin:0;">
      <img src="/images/ml_with_heir/frontend_flow.svg" alt="ML frontend flow from PyTorch, TensorFlow, and ONNX into HEIR's linalg entry level" />
      <figcaption>
        PyTorch, TensorFlow, and ONNX converge into HEIR's linalg entry level
        through torch-mlir, onnx-mlir, and StableHLO.
      </figcaption>
    </figure>
    <div>
      <h3>Linalg Entry Level</h3>
      <p>
        Torch models are converted with <code>torch-mlir</code> to <code>linalg</code> on tensors (with <code>tensor</code> and <code>arith</code> dialects) as HEIR input.
      </p>
      <p>
        The <code>linalg</code> dialect is a funnel dialect for HEIR's
        MLIR frontend. Its abstraction level is required
        for matching on ML kernel operations for
        optimization. Canonicalization patterns
        simplify and reduce memory shuffling
        operations and reduce non-linear operations
        at this level.
      </p>
    </div>
  </div>

<div class="mlheir-card mlheir-card-sidebar" style="margin: 1rem 0 1.5rem;">
  <div>
    <h3>Compilation Configuration</h3>
    <ul>
      <li><strong>Backend</strong> and <strong>scheme</strong> selection</li>
      <li><strong>Secret input</strong> data selected with annotations</li>
      <li>Non-linear activation <strong>approximation</strong> degree</li>
      <li><strong>Range bounds</strong> from model metadata</li>
      <li>User controlled <strong>kernel selection</strong></li>
    </ul>
  </div>
  <div>{{< highlight llvm >}}
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
{{< /highlight >}}</div>
</div>
  <div class="mlheir-band mlheir-band-gold">
    <h2 id="ml-compilation-pipeline" class="mlheir-band-heading">
      ML Compilation Pipeline
    </h2>
  </div>
<div class="mlheir-card mlheir-layout-card mlheir-card-main-sidebar">
<div>
<h3>Relation-based Ciphertext Layouts</h3>
<p>
Layouts are a <em>partial function</em> mapping from the index set of a
cleartext tensor to the index set of a list of ciphertext slots using
<em>Presburger relations</em> and quasi-affine formulas.
</p>
<ul>
<li><strong>Fully general layout annotations</strong> describing plaintext-ciphertext relation</li>
<li><strong>Polyhedral optimization</strong> with Integer Set Library analyzes and manipulates layouts, for e.g. to compute kernel simplifications or slot utilization for batching</li>
</ul>
<p>One useful example maps an <code>(i, j)</code> index in an <code>8 x 8</code> tensor to eight ciphertexts with 1024 slots:</p>
<div class="mlheir-setbuilder">
  <svg class="mlheir-setbuilder-brace" viewBox="0 0 10 100" preserveAspectRatio="none" aria-hidden="true"><path d="M7,3 C4,3 3,6 3,12 L3,43 C3,47 1,48 1,50 C1,52 3,53 3,57 L3,88 C3,94 4,97 7,97" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/></svg>
  <span class="mlheir-setbuilder-map">(i, j) ↦ (ct, slot)</span>
  <span class="mlheir-setbuilder-divider"></span>
  <div class="mlheir-setbuilder-conditions">
    <span>(i − j + ct) &ensp;mod &ensp;8 = 0</span>
    <span>(i − slot) &ensp;mod &ensp;1024 = 0</span>
    <span>0 &lt;= i, j, ct &lt; 8</span>
    <span>0 &lt;= slot &lt; 1024</span>
  </div>
  <svg class="mlheir-setbuilder-brace" viewBox="0 0 10 100" preserveAspectRatio="none" aria-hidden="true"><path d="M3,3 C6,3 7,6 7,12 L7,43 C7,47 9,48 9,50 C9,52 7,53 7,57 L7,88 C7,94 6,97 3,97" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/></svg>
</div>
<p class="mlheir-setbuilder-caption">Mapping (i, j) of an 8×8 tensor to 8 ciphertexts with 1024 slots</p>
</div>
<div>
<figure class="mlheir-figure mlheir-svg-figure" style="margin:0;">
  <img src="/images/ml_with_heir/orion_conv.svg" alt="Orion convolution data layout showing input-filter convolution and resulting diagonal layout matrix" />
  <figcaption>
    <p class="mlheir-caption">Diagram modified from Fig 3 of Orion: A Fully Homomorphic Encryption Framework for Deep Learning</p>
    <div class="mlheir-eq mlheir-eq-plain" style="margin:0;">
      <div class="mlheir-eq-line" style="font-size:1.3rem;">m<sub>r</sub> = (i<sub>dr</sub> + P)F + i<sub>dc</sub> + P</div>
      <div class="mlheir-eq-line" style="font-size:1.3rem;">m<sub>c</sub> = W<sub>d</sub>i<sub>dr</sub> + i<sub>dc</sub> + W<sub>d</sub>i<sub>fr</sub> + i<sub>fc</sub></div>
    </div>
  </figcaption>
</figure>
</div>
</div>
<div class="mlheir-card" style="margin-top:1.25rem;">
<h3>Layout Optimization Flow</h3>
<div class="mlheir-card-auto-content">
<div style="display:flex; flex-direction:column; gap:0.5rem; align-items:center;">
  <div class="mlheir-flow-chip" style="width:100%; text-align:center; padding:0.5rem 0.8rem; border-radius:0.6rem;">Propagate</div>
  <p style="margin:0; font-size:0.85rem; text-align:center;">Forward analysis propagates IR with default layouts and kernels.</p>
  <span style="color:var(--mlheir-blue); font-size:1.2rem;">↓</span>
  <div class="mlheir-flow-chip" style="width:100%; text-align:center; padding:0.5rem 0.8rem; border-radius:0.6rem;">Optimize</div>
  <p style="margin:0; font-size:0.85rem; text-align:center;">Cost models select optimal kernels to minimize cost and layout conversions.</p>
  <span style="color:var(--mlheir-blue); font-size:1.2rem;">↓</span>
  <div class="mlheir-flow-chip" style="width:100%; text-align:center; padding:0.5rem 0.8rem; border-radius:0.6rem;">Simplify</div>
  <p style="margin:0; font-size:0.85rem; text-align:center;">Backwards traversal hoists layout conversions to encodings.</p>
</div>
<div>
<h4>New Layout Integrations</h4>
<p>
HEIR integrates
<a href="https://doi.org/10.1109/TIFS.2024.3490862">bicyclic</a> [8]
and
<a href="https://eprint.iacr.org/2025/1200">tricyclic</a> [9]
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
  <div class="mlheir-card mlheir-card-two-col" style="margin-top: 1.25rem;">
    <div>
      <h3>Optimization Variety Pack</h3>
      <p>HEIR's ML pipeline utilizes a number of generally applicable optimization patterns:</p>
      <ul>
        <li><strong>Sparse</strong> matrix product simplification</li>
        <li><strong>Baby-step giant-step</strong> for general reductions</li>
        <li><strong>Minimal depth</strong> polynomials evaluation with Paterson-Stockmeyer</li>
        <li><strong>Fast</strong> (hoisted) rotation rewrites</li>
        <li><strong>Minimized extended</strong> key basis switching</li>
        <li><strong>High level</strong> program vectorization</li>
        <li><strong>Shift networks</strong> for layout conversions</li>
        <li><strong>Loop support</strong> with HALO optimizations</li>
        <li><strong>Multiplexed data packing</strong> for slot utilization</li>
      </ul>
    </div>
    <div class="mlheir-transform-layout">
      <div class="mlheir-transform-inner">
        <div class="mlheir-transform-ladder">
          <div class="mlheir-transform-step">Model Transforms</div>
          <div class="mlheir-transform-arrow">↓</div>
          <div class="mlheir-transform-step">Arithmetization</div>
          <div class="mlheir-transform-arrow">↓</div>
          <div class="mlheir-transform-step">Vectorization</div>
          <div class="mlheir-transform-arrow">↓</div>
          <div class="mlheir-transform-step">Layout Pipeline</div>
          <div class="mlheir-transform-arrow">↓</div>
          <div class="mlheir-transform-step">Noise Management</div>
          <div class="mlheir-transform-arrow">↓</div>
          <div class="mlheir-transform-step">Parameter Selection</div>
        </div>
      </div>
      <div class="mlheir-transform-side">
        <div class="mlheir-transform-side-spacer" style="flex: 3"></div>
        <div class="mlheir-transform-side-item">
          <div>Plaintext Execution</div>
        </div>
        <div class="mlheir-transform-side-spacer" style="flex: 1"></div>
        <div class="mlheir-transform-side-item">
          <div>Scheme IR</div>
        </div>
      </div>
      <div class="mlheir-transform-caption">Transforms operate in the linalg dialect to secret arithmetic</div>
    </div>
  </div>

<div class="mlheir-card mlheir-card-sidebar" style="margin: 1rem 0 1.5rem;">
  <div>
    <h3>Make It Easy</h3>
    <p>HEIR simplifies the developer and debugging experience with:</p>
    <ul>
      <li><strong>Tracking and debugging</strong> utilities from MLIR</li>
      <li><strong>Plaintext</strong> execution mode with custom debug handlers</li>
      <li><strong>Client</strong> helpers for encoding and encryption/decryption</li>
      <li>Output code is <strong>human-readable</strong> code to support inspection and modification</li>
      <li><strong>Cleartext</strong> computations are hoisted to separate functions for precomputation</li>
      <li><strong>Scheme-specific</strong> parameter selection</li>
    </ul>
  </div>
  <div>{{< highlight cpp >}}
// preprocessing functions
PlaintextT matvec__preprocessing(CryptoContextT cc) {
  ...
  const auto& pt2 = cc->MakeCKKSPackedPlaintext(c0);
  return pt2;
}

// main workload
CiphertextT matvec(CryptoContextT cc, CiphertextT ct) {
  ...
  const auto& ct5 = cc->EvalMult(ct4, pt2);
  const auto& ct6 = cc->EvalRotate(ct, 3);
  ...
  const auto& ct47 = cc->EvalAdd(ct38, ct46);
  const auto& ct48 = cc->EvalMultNoRelin(ct47, ct47);
  const auto& ct49 = cc->Relinearize(ct48);
  ...
}

// client functions
CiphertextT matvec__encrypt__arg0(
    CryptoContextT cc, std::vector<float> v0, PublicKeyT pk);
std::vector<float> matvec__decrypt__result0(
    CryptoContextT cc, CiphertextT ct, PrivateKeyT sk);
CryptoContextT matvec__generate_crypto_context();
CryptoContextT matvec__configure_crypto_context(
    CryptoContextT cc, PrivateKeyT sk);
{{< /highlight >}}</div>

</div>
  <div class="mlheir-band mlheir-band-red">
    <h2 id="hardware-integrations" class="mlheir-band-heading">
      Hardware Integrations
    </h2>
  </div>
  <div class="mlheir-grid mlheir-grid-two">
    <div class="mlheir-card mlheir-flat">
      <div class="mlheir-layer-stack-wrap">
      <div class="mlheir-layer-stack">
        <div class="mlheir-token-row">
          <span class="mlheir-token mlheir-token-amber mlheir-token-standalone">Python</span>
          <span class="mlheir-token mlheir-token-amber mlheir-token-standalone">Torch</span>
          <span class="mlheir-token mlheir-token-amber mlheir-token-standalone">TensorFlow Lite</span>
        </div>
        <div class="mlheir-layer mlheir-layer-gray">
          <strong>Standard MLIR</strong>
          <div class="mlheir-token-row">
            <span class="mlheir-token mlheir-token-gray">func</span>
            <span class="mlheir-token mlheir-token-gray">linalg</span>
            <span class="mlheir-token mlheir-token-gray">tensor</span>
            <span class="mlheir-token mlheir-token-gray">arith</span>
            <span class="mlheir-token mlheir-token-gray">affine</span>
            <span class="mlheir-token mlheir-token-gray">...</span>
          </div>
        </div>
        <div class="mlheir-layer mlheir-layer-green">
          <strong>Secret arithmetic</strong>
          <div class="mlheir-token-row">
            <span class="mlheir-token mlheir-token-green">secret</span>
            <span class="mlheir-token mlheir-token-green">tensor_ext</span>
            <span class="mlheir-token mlheir-token-green">mgmt</span>
            <span class="mlheir-token mlheir-token-green">polynomial</span>
            <span class="mlheir-token mlheir-token-green">comb</span>
          </div>
        </div>
        <div class="mlheir-layer mlheir-layer-blue">
          <strong>Scheme APIs</strong>
          <div class="mlheir-token-row">
            <span class="mlheir-token mlheir-token-blue">lwe</span>
            <span class="mlheir-token mlheir-token-blue">bgv</span>
            <span class="mlheir-token mlheir-token-blue">ckks</span>
            <span class="mlheir-token mlheir-token-blue">cggi</span>
          </div>
        </div>
        <div class="mlheir-stack-split">
          <div class="mlheir-stack-split-left">
            <div class="mlheir-layer mlheir-layer-violet">
              <strong>Scheme implementation</strong>
              <div class="mlheir-token-row">
                <span class="mlheir-token mlheir-token-violet">polynomial</span>
                <span class="mlheir-token mlheir-token-violet">rns</span>
                <span class="mlheir-token mlheir-token-violet">mod_arith</span>
              </div>
            </div>
            <div class="mlheir-layer mlheir-layer-red">
              <strong>Hardware dialects</strong>
              <div class="mlheir-token-row">
                <span class="mlheir-token mlheir-token-red">llvm</span>
                <span class="mlheir-token mlheir-token-red">scifr</span>
                <span class="mlheir-token mlheir-token-red">...</span>
              </div>
            </div>
          </div>
          <div class="mlheir-layer mlheir-layer-red">
            <strong>Library APIs</strong>
            <div class="mlheir-token-row">
              <span class="mlheir-token mlheir-token-red">lattigo</span>
              <span class="mlheir-token mlheir-token-red">tfhe_rust</span>
              <span class="mlheir-token mlheir-token-red">jaxite</span>
              <span class="mlheir-token mlheir-token-red">openfhe</span>
            </div>
          </div>
        </div>
      </div>
      <div class="mlheir-layer-stack-arrow"></div>
      </div>
    </div>
    <div>
      <div class="mlheir-card" style="height: 100%; box-sizing: border-box;">
        <h3>Exit Dialects</h3>
        <p>
          Support for multiple backends (CPU, GPU, FPGA, ASICs, and photonics)
          allows for comprehensive testing and benchmarking. After HEIR's high
          level program analysis and compilation, data layouts, kernels,
          schemes, and parameters are selected and the IR uses scheme level
          operations. Scheme level IR is lowered in two possible ways
          to exit HEIR:
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
    </div>
  </div>
  <div class="mlheir-grid mlheir-vendor-grid">
    <div class="mlheir-card mlheir-logo-card">
      <img src="/images/ml_with_heir/optalysys.jpg" alt="Optalysys logo" />
      <p>
        Optalysys utilizes <em>photonic computing technology</em> to
        perform modular arithmetic operations over the Polynomial Modular
        Number System (PMNS). Integration with HEIR's generated low level
        NTT and mod arith code will allow running FHE workloads on
        Optalysys' optical processing chips.
      </p>
      <p style="font-size:0.8rem; color:var(--mlheir-muted); margin-top:auto;">
        <a href="https://optalysys.com/resource/optalysys-partners-with-google-heir/">optalysys.com/resource/optalysys-partners-with-google-heir</a>
      </p>
    </div>
    <div class="mlheir-card mlheir-logo-card">
      <img src="/images/ml_with_heir/belfort.svg" alt="Belfort logo" />
      <p>
        Belfort integrates their FPGA-based accelerator with HEIR
        through the CGGI boolean and shortint APIs. They utilize
        vectorization strategies in HEIR and software optimizations in
        their custom <code>tfhe-rs</code> library for performance.
      </p>
      <p style="font-size:0.8rem; color:var(--mlheir-muted); margin-top:auto;">
        Ian Berkenstein, Milind et al. "<a href="https://dl.acm.org/doi/10.1145/3470496.3527415">BTS: An Accelerator for Bootstrappable Fully
        Homomorphic Encryption</a>," in <em>Proceedings of the 49th Annual International Symposium on Computer Architecture</em>,
        ACM, 2022.
      </p>
    </div>
    <div class="mlheir-card mlheir-logo-card">
      <img src="/images/ml_with_heir/cornami.jpg" alt="Cornami logo" />
      <p>
        Cornami's MX2 systolic array is integrated as a backend to HEIR's
        MLIR pipeline for CGGI and CKKS schemes. HEIR exits to Cornami's
        Secure Computing Interface Framework (SCIFR) with custom
        optimizations.
      </p>
      <p style="font-size:0.8rem; color:var(--mlheir-muted); margin-top:auto;">
        Custozimov, Denis et al. "<a href="https://arxiv.org/abs/2510.16025">Resource-Sensitive Integration of CGGI and CKKS schemes
        on the Cornami Computing Target</a>," <em>ArXiv</em>, 2025.
      </p>
    </div>
    <div class="mlheir-card mlheir-logo-card">
      <div class="mlheir-wordmark">CROSS</div>
      <p>
        TPU-native CKKS implementation with SoTA performance vs GPU (20ms
        bootstrap) using JAX. HEIR integration utilizes the CKKS dialect
        to lower to the CROSS API exit dialect.
      </p>
      <p style="font-size:0.8rem; color:var(--mlheir-muted); margin-top:auto;">
        Fang, Jiangteng et al. "<a href="https://arxiv.org/abs/2501.07047">Leveraging ASIC AI Chips for Homomorphic Encryption</a>," in
        <em>IEEE International Symposium on High-Performance Computer Architecture</em>, 2025.
      </p>
    </div>
    <div class="mlheir-card mlheir-logo-card">
      <img src="/images/ml_with_heir/fhetch.png" alt="FHE Tech logo" />
      <p>
        HEIR tracks progress of the <em>polynomial intermediate
        representation</em> (IR) developed by FHE Technical Consortium for
        Hardware (FHETCH). The IR aims to provide a standardised set of
        hardware-level operations for interoperable platform integration.
        HEIR's <em>polynomial</em> dialect aligns with the evolving standard.
      </p>
      <p style="font-size:0.8rem; color:var(--mlheir-muted); margin-top:auto;">The global FHE hardware consortium (<a href="https://www.fhetch.org">www.fhetch.org</a>)</p>
    </div>
    <div class="mlheir-card mlheir-logo-card mlheir-logo-note">
      <p>
        Plus more backends in progress (e.g. FIDESlib GPU backend) and
        under NDA.
      </p>
      <p style="font-size:0.8rem; color:var(--mlheir-muted); margin-top:auto;">
        C. Aguilo-Domingo et al., "<a href="https://ieeexplore.ieee.org/document/11096404">FIDESlib: A Fully-Fledged Open-Source
        FHE Library for Efficient CKKS on GPUs</a>," in 2025 IEEE
        International Symposium on Performance Analysis of Systems and
        Software (ISPASS), Ghent, Belgium, 2025, pp. 1-3.
      </p>
    </div>
  </div>
  <div class="mlheir-band mlheir-band-green">
    <h2 id="community" class="mlheir-band-heading">
      Community
    </h2>
  </div>
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
          <svg class="mlheir-linkicon" viewBox="0 0 24 24" fill="currentColor"><path d="M12 .3a12 12 0 00-3.8 23.38c.6.11.82-.26.82-.58v-2.16c-3.34.73-4.04-1.61-4.04-1.61a3.18 3.18 0 00-1.33-1.76c-1.09-.74.08-.73.08-.73a2.52 2.52 0 011.84 1.24 2.56 2.56 0 003.5 1 2.56 2.56 0 01.76-1.6c-2.67-.3-5.47-1.33-5.47-5.93a4.64 4.64 0 011.24-3.22 4.3 4.3 0 01.12-3.18s1-.32 3.3 1.23a11.38 11.38 0 016 0c2.28-1.55 3.28-1.23 3.28-1.23a4.3 4.3 0 01.12 3.18 4.64 4.64 0 011.24 3.22c0 4.61-2.81 5.63-5.48 5.93a2.86 2.86 0 01.82 2.22v3.29c0 .32.21.7.82.58A12 12 0 0012 .3"/></svg>
          <strong>GitHub</strong>
          <span>github.com/google/heir</span>
        </a>
        <a class="mlheir-linkbox" href="https://discord.com/invite/fhe-org">
          <svg class="mlheir-linkicon" viewBox="0 0 24 24" fill="currentColor"><path d="M20.32 4.37A19.8 19.8 0 0015.39 3a13.2 13.2 0 00-.62 1.24 18.5 18.5 0 00-5.54 0A12.5 12.5 0 008.6 3 19.7 19.7 0 003.68 4.37 20.27 20.27 0 00.1 17.73a19.9 19.9 0 006.07 3.06 14.7 14.7 0 001.29-2.1 12.9 12.9 0 01-2.03-.97l.49-.38a14.18 14.18 0 0012.16 0l.49.38c-.65.38-1.33.7-2.03.97.37.73.8 1.43 1.29 2.1a19.85 19.85 0 006.07-3.06A20.22 20.22 0 0020.32 4.37zM8.02 15.07c-1.18 0-2.16-1.08-2.16-2.42s.96-2.42 2.16-2.42 2.18 1.08 2.16 2.42c0 1.34-.96 2.42-2.16 2.42zm7.96 0c-1.18 0-2.16-1.08-2.16-2.42s.96-2.42 2.16-2.42 2.18 1.08 2.16 2.42c0 1.34-.95 2.42-2.16 2.42z"/></svg>
          <strong>Discord</strong>
          <span>discord.com/invite/fhe-org</span>
        </a>
        <a class="mlheir-linkbox" href="/community/">
          <svg class="mlheir-linkicon" viewBox="0 0 24 24" fill="currentColor"><path d="M16 11c1.66 0 2.99-1.34 2.99-3S17.66 5 16 5c-1.66 0-3 1.34-3 3s1.34 3 3 3zm-8 0c1.66 0 2.99-1.34 2.99-3S9.66 5 8 5C6.34 5 5 6.34 5 8s1.34 3 3 3zm0 2c-2.33 0-7 1.17-7 3.5V19h14v-2.5c0-2.33-4.67-3.5-7-3.5zm8 0c-.29 0-.62.02-.97.05 1.16.84 1.97 1.97 1.97 3.45V19h6v-2.5c0-2.33-4.67-3.5-7-3.5z"/></svg>
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
        <li>Integrating <strong>Gentry-Lee FHE scheme</strong> "Fully Homomorphic Encryption for Matrix Arithmetic"</li>
        <li><strong>Layout optimizer</strong> that uses the structure of Presburger relations, and/or the general joint layout+kernel selection problem</li>
        <li>New <strong>FHE scheme implementations</strong> (e.g. GBFV) and optimizations</li>
        <li>Incorporating <strong>memory constraints</strong> into cost models for kernel compilation</li>
        <li><strong>Profile-guided optimizations</strong> for parameter selection &amp; scale management</li>
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
  <h2>References</h2>
  <ol>
    <li>E. Chen et al.,
      <a href="https://eprint.iacr.org/2025/1319">Bridging Usability and Performance: A Tensor Compiler for Autovectorizing Homomorphic Encryption</a>,
      <em>IACR Cryptol. ePrint Arch.</em>, 2025/1319.</li>
    <li>Z. Zhou et al.,
      <a href="https://eprint.iacr.org/2026/213">Orbit: Optimizing Rescale and Bootstrap Placement with Integer Linear Programming Techniques for Secure Inference</a>,
      <em>Cryptology ePrint Archive</em>, 2026/213.</li>
    <li>E. Ünay et al.,
      <a href="https://doi.org/10.48550/arXiv.2601.18445">KeyMemRT Compiler and Runtime: Unlocking Memory-Scalable FHE</a>,
      <em>arXiv:2601.18445</em>, 2026.</li>
    <li>M. Gao and H. Zheng,
      <a href="https://eprint.iacr.org/2025/1036">A Critique on Average-Case Noise Analysis in RLWE-Based Homomorphic Encryption</a>,
      <em>Proceedings of the 13th Workshop on Encrypted Computing &amp; Applied Homomorphic Computing</em>, 2025.</li>
    <li>A. Krastev et al.,
      <a href="https://doi.org/10.1145/3656382">A Tensor Compiler with Automatic Data Packing for Simple and Efficient Fully Homomorphic Encryption</a>,
      <em>Proceedings of the ACM on Programming Languages</em>, vol. 8, no. PLDI, June 2024, pp. 126–50.</li>
    <li>S. Cheon et al.,
      <a href="https://doi.org/10.1145/3669940.3707275">HALO: Loop-Aware Bootstrapping Management for Fully Homomorphic Encryption</a>,
      <em>Proceedings of the 30th ACM International Conference on Architectural Support for Programming Languages and Operating Systems</em>, Volume 1, 2025, pp. 572–85.</li>
    <li>A. Ebel et al.,
      <a href="https://doi.org/10.1145/3676641.3716008">Orion: A Fully Homomorphic Encryption Framework for Deep Learning</a>,
      <em>Proceedings of the 30th ACM International Conference on Architectural Support for Programming Languages and Operating Systems</em>, Volume 2, 2025, pp. 734–49.</li>
    <li>J. Chen, L. Yang, W. Wu, Y. Liu, and Y. Feng,
      <a href="https://doi.org/10.1109/TIFS.2024.3490862">Homomorphic Matrix Operations Under Bicyclic Encoding</a>,
      <em>IEEE Transactions on Information Forensics and Security</em>, vol. 20, 2025, pp. 1390–404.</li>
    <li>L. Lim et al.,
      <a href="https://eprint.iacr.org/2025/1200">Tricycle: Private Transformer Inference with Tricyclic Encodings</a>,
      <em>Cryptology ePrint Archive</em>, 2025/1200.</li>
    <li>J. Vos et al.,
      <a href="https://doi.org/10.1007/978-3-031-17140-6_20">Efficient Circuits for Permuting and Mapping Packed Values Across Leveled Homomorphic Ciphertexts</a>,
      <em>Computer Security – ESORICS 2022</em>, Springer, 2022, pp. 408–23.</li>
  </ol>
</div>
</article>
<!-- mdformat global-off -->
