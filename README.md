# net-gdr-reproducer
## Installing dependencies

```
bash create_env.sh
```

## Running

The following needs to be done once.

```
chmod +x get_rank_from_slurm.sh
```


Then run

```
sbatch -N 32 -A <project> run.sh
```
