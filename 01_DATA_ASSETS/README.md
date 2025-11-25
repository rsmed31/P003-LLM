# Postgres

Start: docker compose up --build  
Stop:  docker compose down -v

Volume layout (pg 18+): mount /var/lib/postgresql (not /var/lib/postgresql/data).

Reset & rebuild if schema or init scripts change:
```
docker compose down -v
docker compose up --build
```

If init script error “cannot execute: required file not found”: ensure LF endings & executable:
```
git config core.autocrlf false
sed -i 's/\r$//' 01_DATA_ASSETS/postgres_api/02-init.sh
chmod +x 01_DATA_ASSETS/postgres_api/02-init.sh
```
