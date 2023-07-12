sbatch(1)                                                                  Slurm Commands                                                                  sbatch(1)

NAME
       sbatch - Submit a batch script to Slurm.

SYNOPSIS
       sbatch [OPTIONS(0)...] [ : [OPTIONS(N)...]] script(0) [args(0)...]

       Option(s) define multiple jobs in a co-scheduled heterogeneous job.  For more details about heterogeneous jobs see the document
       https://slurm.schedmd.com/heterogeneous_jobs.html

DESCRIPTION
       sbatch  submits  a  batch script to Slurm.  The batch script may be given to sbatch through a file name on the command line, or if no file name is specified,
       sbatch will read in a script from standard input. The batch script may contain options preceded with "#SBATCH" before any executable commands in the  script.
       sbatch will stop processing further #SBATCH directives once the first non-comment non-whitespace line has been reached in the script.

       sbatch exits immediately after the script is successfully transferred to the Slurm controller and assigned a Slurm job ID.  The batch script is not necessar-
       ily granted resources immediately, it may sit in the queue of pending jobs for some time before its required resources become available.

       By default both standard output and standard error are directed to a file of the name "slurm-%j.out", where the "%j" is replaced with the job allocation num-
       ber. The file will be generated on the first node of the job allocation.  Other than the batch script itself, Slurm does no movement of user files.

       When  the  job  allocation  is  finally  granted for the batch script, Slurm runs a single copy of the batch script on the first node in the set of allocated
       nodes.

       The following document describes the influence of various options on the allocation of cpus to jobs and tasks.
       https://slurm.schedmd.com/cpu_management.html

RETURN VALUE
       sbatch will return 0 on success or error code on failure.

SCRIPT PATH RESOLUTION
       The batch script is resolved in the following order:

       1. If script starts with ".", then path is constructed as: current working directory / script
       2. If script starts with a "/", then path is considered absolute.
       3. If script is in current working directory.
       4. If script can be resolved through PATH. See path_resolution(7).

       Current working directory is the calling process working directory unless the --chdir argument is passed, which will override the current working directory.

OPTIONS
       -A, --account=<account>
              Charge resources used by this job to specified account.  The account is an arbitrary string. The account name may be changed after job submission  us-
              ing the scontrol command.

       --acctg-freq=<datatype>=<interval>[,<datatype>=<interval>...]
              Define  the  job  accounting  and  profiling  sampling intervals in seconds.  This can be used to override the JobAcctGatherFrequency parameter in the
 Manual page sbatch(1) line 1/1682 3% (press h for help or q to quit)
