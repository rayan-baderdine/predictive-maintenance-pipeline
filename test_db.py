from sqlalchemy import create_engine, text

engine = create_engine('postgresql://postgres:0000@localhost:5432/bearing_db')

with engine.connect() as conn:
    count = conn.execute(text('SELECT COUNT(*) FROM raw_readings')).scalar()
    print('Total rows in database:', count)
    
    rows = conn.execute(text('SELECT * FROM raw_readings LIMIT 3'))
    for row in rows:
        print(row)