terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 4.0"
    }
  }
}
# Configure the AWS Provider
provider "aws" {
  shared_config_files      = ["~/.aws/config"]
  region                   = "us-west-1"
  shared_credentials_files = ["~/.aws/credentials"]
  profile                  = "default"
}
