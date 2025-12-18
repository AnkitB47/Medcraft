variable "resource_group_name" {
  description = "The name of the resource group"
  type        = string
  default     = "medcraft-rg"
}

variable "location" {
  description = "The Azure region to deploy to"
  type        = string
  default     = "eastus"
}
