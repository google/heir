# HEIR on Google cloud TPU

This document provides a brief introduction to running FHE programs compiled
with [HEIR](https://heir.dev) on cloud TPU. Jaxite library is used to run
programs on TPU.

Warning: You will be charged for the TPU. Please ensure TPU VMs are stopped
after use.

## One time setup

One time setup is required to run the tool.

### Setting up the GCP project

Before you follow this quickstart you must create a Google Cloud Platform
account, install the Google Cloud CLI and configure the `gcloud` command. For
more information see
[Set up an account and a Cloud TPU project](https://cloud.google.com/tpu/docs/setup-gcp-account)

### Install the Google Cloud CLI

The Google Cloud CLI contains tools and libraries for interfacing with Google
Cloud products and services. For more information see
[Installing the Google Cloud CLI](https://cloud.google.com/sdk/docs/install)

#### Configure the gcloud command

Run the following commands to configure `gcloud` to use your Google Cloud
project.

```sh
$ gcloud config set account your-email-account
$ gcloud config set project your-project-id
```

### Enable the cloud TPU API

Enable the Cloud TPU API and create a service identity.

```sh
$ gcloud services enable tpu.googleapis.com
$ gcloud beta services identity create --service tpu.googleapis.com
```

### Provision a TPU

- Clone HEIR repo and install dependencies

```sh
$ git clone https://github.com/google/heir.git
$ cd heir
```

Optionally create a python virtual environment for cloud dependencies.

```sh
$ python3 -m venv fhe_cloud
$ source fhe_cloud/bin/activate
```

Note: To deactivate, run the command "deactivate".

```sh
$ pip install -r ./scripts/gcp/requirements.txt
```

- Create a TPU

Create a new TPU called "heirtpu4" with 4 tpu devices and the required
infrastructure.

```sh
$ ./scripts/gcp/tool provision heirtpu4 --zone="us-south1-a"
```

## Execute FHE programs on the TPU

The following section provides a few programs on run on cloud TPU. The cloud
tool starts a VM copies the required files onto the heirtpu4 VM, executes them
and stops the VM. Optional: Use the flag --keep_running to keep the TPU running.
In this case, please remember to stop the VM on cloud console.

### Execute a single CGGI and-gate

Execute a basic and-gate on TPU using jaxite

```sh
$ ./scripts/gcp/tool run \
--vm="heirtpu4" \
--zone="us-south1-a" \
--files="./scripts/gcp/examples/jaxite_example.py" \
--main="./scripts/gcp/examples/jaxite_example.py"
```

The above program will display the timing metric. of ~8ms for a single and-gate
bootstrap on jaxite.

### Execute a string demo to redact ssn

Execute a basic and-gate on TPU using jaxite

```sh
$ ./scripts/gcp/tool run \
--vm="heirtpu4" \
--zone="us-south1-a" \
--files="./scripts/gcp/examples/redact_ssn_fhe_py_lib_p1.py" \
--main="./scripts/gcp/examples/redact_ssn_main.py"
```

The above program will display the timing metric. of ~8ms for a single and-gate
bootstrap on jaxite.

## Execute a HEIR compiled FHE program on the TPU

Compile a HEIR program and run on TPU Run the following commands to compile a
program in HEIR.

```sh
$ bazel build scripts/gcp/examples:add_one_lut3
```

Run the following command to run the above compiled program on TPU.

```sh
$ ./scripts/gcp/tool run \
--vm="heirtpu4" \
--zone="us-south1-a" \
--files="./bazel-bin/scripts/gcp/examples/add_one_lut3_lib.py" \
--main="./scripts/gcp/examples/add_one_lut3_main.py"
```

## Pricing

See [Pricing](https://cloud.google.com/tpu/pricing) for more details. Though
stopped TPU VM does not incur any cost, the disk attached to the VM does. The
cost is the same as the cost of the disk when the VM is running. See
[Disk and image pricing](https://cloud.google.com/compute/disks-image-pricing#disk)
for more details.
