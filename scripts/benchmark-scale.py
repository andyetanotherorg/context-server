#!/usr/bin/env python3
"""Generate repeatable structural search benchmarks at target chunk counts."""
import argparse, os, shutil, sqlite3, subprocess, time

p=argparse.ArgumentParser()
p.add_argument('--source-db', required=True)
p.add_argument('--binary', default='./target/release/context-server')
p.add_argument('--counts', default='1000,10000,50000')
p.add_argument('--query', default='password reset')
a=p.parse_args()
for target in map(int,a.counts.split(',')):
    db=f'/tmp/context-server-bench-{target}.db'
    shutil.copy2(a.source_db,db)
    con=sqlite3.connect(db)
    docs=con.execute('select source_path,chunk_index,text,headings,metadata from documents').fetchall()
    vecs=[r[0] for r in con.execute('select vector from embeddings order by id')]
    con.execute('delete from embeddings'); con.execute('delete from documents'); con.execute('delete from files')
    con.executemany('insert into documents(id,source_path,chunk_index,text,headings,metadata) values(?,?,?,?,?,?)',((i+1,f'copy-{i//len(docs):06d}/{d[0]}',d[1],d[2],d[3],d[4]) for i in range(target) for d in [docs[i%len(docs)]]))
    con.executemany('insert into embeddings(id,dim,vector) values(?,?,?)',((i+1,384,vecs[i%len(vecs)]) for i in range(target)))
    con.commit(); con.close()
    start=time.perf_counter(); r=subprocess.run([a.binary,'search','--db',db,a.query],stdout=subprocess.DEVNULL); elapsed=time.perf_counter()-start
    print(f'{target}\t{os.path.getsize(db)/1048576:.1f} MiB\t{elapsed:.3f}s\texit={r.returncode}')
    os.unlink(db)
