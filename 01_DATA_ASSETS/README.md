# Postgres

Start: sudo docker compose up --build
Stop:  sudo docker compose down -v

## Note on SQL changes

Once the database is initialized, PostgreSQL stores the data in volumes. If you make changes to the SQL init files, they won’t be applied automatically on the next start.

To apply updated SQL changes, you must reset the volumes:
with sudo docker compose down -v
