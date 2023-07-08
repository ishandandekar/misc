## Terraform notes

### The General Sequence of Terraform Commands:

- `terraform init`: Initializes your project
- `terraform plan`: Checks your configuration against the current state and generates a plan
- `terraform apply`: Applies the plan to create or update your infrastructure
- `terraform destroy`: Removes resources when no longer needed

### Terraform provider architecture

![architecure](https://courses.devopsdirective.com/_next/image?url=%2Fterraform-beginner-to-pro%2F02-01-tf-architecture.jpg&w=1080&q=75)

### Terraform Providers:

- Visit _registry.terraform.io_ to explore available providers
- Official providers have the "official" tag and are maintained by the respective cloud service
- In your configuration file, specify required providers and pin their versions within a terraform block

- Use `--auto-approve` argument to not type "yes" every time.

- `terraform state list`: Used to list all the states (services).

- The **`terraform.tfvars`** has precedance over the `variables.tf` file. Which means, the values in `variables.tf` will be overwritten by `terraform.tfvars`
- The flag **`-var=""`** overwrites the variables even if they are present in `terraform.tfvars` as well as `variables.tf`. Similarly, you can use **`-var-file=""`** to specify the vars file to be used.
