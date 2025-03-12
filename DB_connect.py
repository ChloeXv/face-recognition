import psycopg2

# 连接 PostgreSQL
conn = psycopg2.connect(
    dbname="media-analysis_dev",
    user="postgres",
    password="a3d20$3Dss",
    host="172.16.52.103",
    port="5432"
)

cursor = conn.cursor()
cursor.execute("SELECT current_database();")
print(cursor.fetchone())

cursor.close()
conn.close()
