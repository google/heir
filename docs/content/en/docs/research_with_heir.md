---
title: Research with HEIR
weight: 45
---

## Citing HEIR

[HEIR: A Universal Compiler for Homomorphic Encryption](https://arxiv.org/abs/2508.11095)

```
@misc{ali2025heir,
      title={HEIR: A Universal Compiler for Homomorphic Encryption},
      author={Asra Ali and Jaeho Choi and Bryant Gipson and Shruthi Gorantala
              and Jeremy Kun and Wouter Legiest and Lawrence Lim and Alexander
              Viand and Meron Zerihun Demissie and Hongren Zheng},
      year={2025},
      eprint={2508.11095},
      archivePrefix={arXiv},
      primaryClass={cs.CR},
      url={https://arxiv.org/abs/2508.11095},
}
```

## Publications built on HEIR

- [Circuit Optimization Using Arithmetic Table Lookups](https://dl.acm.org/doi/10.1145/3729258)
  (PLDI 2025). Raghav Malik, Vedant Paranjape, Milind Kulkarni.
- [Resource Estimation of CGGI and CKKS scheme workloads on FracTLcore Computing Fabric](https://arxiv.org/abs/2510.16025).
  Denis Ovichinnikov, Hemant Kavadia, Satya Keerti Chand Kudupudi, Ilya Rempel,
  Vineet Chadha, Marty Franz, Paul Master, Craig Gentry, Darlene Kindler,
  Alberto Reyes, Muthu Annamalai.
- [A Critique on Average-Case Noise Analysis in RLWE-Based Homomorphic Encryption](https://dl.acm.org/doi/10.1145/3733811.3767312)
  (WAHC '25). Mingyu Gao, Hongren Zheng.

## Doing private research using HEIR

Our project will be developed in an open-source GitHub repository. If you'd like
to work on a non-public branch while still accessing the latest developments, we
recommend the following setup. This process will result in two remote
repositories: one public for submitting pull requests (PRs) to the original repo
and one private.

1. **Fork the Repository**: Fork the
   [google/heir](https://github.com/google/heir) repo to a public fork on your
   GitHub repository. This should create a project at
   `https://github.com/<username>/heir`

1. **Create a Private Repository**: Create a new private repo using the
   [GitHub UI](https://github.com/new), e.g. named `heir-private`

1. **Link Your Fork to the Private Repository** Couple your fork to the new
   private repo

```bash
git clone --bare git@github.com:<username>/heir.git heir-public
cd heir-public
git push --mirror git@github.com:<username>/heir-private.git
cd ..
rm -rf heir-public
```

4. **Clone the Private Repository** Now, you can clone the private repo to work
   locally

```bash
git clone git@github.com:<username>/heir-private.git
cd heir-private
```

5. **Add the Private Repository as a Remote to Your Public Repository**
   Additionally, you can add the private repo as a remote target to your public
   repo. This way, the private branch will be locally available, while you can
   push commits to the private repo.

```bash
cd heir
git remote add private git@github.com:<username>/heir-private.git
git fetch --all
git checkout private/new_branch
```

Note that using `git push private new_branch2` will push the commit/branch to
the private repo.

Once you're ready to publish your development work, you can push your commits to
a branch in the public repository and create a pull request.

<!-- mdformat global-off -->
