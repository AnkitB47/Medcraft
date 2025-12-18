provider "azurerm" {
  features {}
}

resource "azurerm_resource_group" "medcraft" {
  name     = var.resource_group_name
  location = var.location
}

resource "azurerm_kubernetes_cluster" "aks" {
  name                = "medcraft-aks"
  location            = azurerm_resource_group.medcraft.location
  resource_group_name = azurerm_resource_group.medcraft.name
  dns_prefix          = "medcraftaks"

  default_node_pool {
    name       = "default"
    node_count = 2
    vm_size    = "Standard_DS2_v2"
  }

  identity {
    type = "SystemAssigned"
  }
}

resource "azurerm_container_registry" "acr" {
  name                = "medcraftacr"
  resource_group_name = azurerm_resource_group.medcraft.name
  location            = azurerm_resource_group.medcraft.location
  sku                 = "Basic"
  admin_enabled       = true
}

resource "azurerm_postgresql_flexible_server" "db" {
  name                   = "medcraft-db"
  resource_group_name    = azurerm_resource_group.medcraft.name
  location               = azurerm_resource_group.medcraft.location
  version                = "13"
  administrator_login    = "medcraftadmin"
  administrator_password = "Password123!" # In production, use KeyVault
  storage_mb             = 32768
  sku_name               = "B_Standard_s1"
}

resource "azurerm_redis_cache" "redis" {
  name                = "medcraft-redis"
  location            = azurerm_resource_group.medcraft.location
  resource_group_name = azurerm_resource_group.medcraft.name
  capacity            = 1
  family              = "C"
  sku_name            = "Basic"
  enable_non_ssl_port = true
}

resource "azurerm_storage_account" "storage" {
  name                     = "medcraftstorage"
  resource_group_name      = azurerm_resource_group.medcraft.name
  location                 = azurerm_resource_group.medcraft.location
  account_tier             = "Standard"
  account_replication_type = "LRS"
}
