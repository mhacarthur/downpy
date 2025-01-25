#!/bin/bash
#SBATCH --job-name=mi_trabajo
#SBATCH --output=output.txt  # Archivo donde se guardará la salida
#SBATCH --ntasks=1           # Número de tareas (en este caso, 1 porque es un solo script)
#SBATCH --cpus-per-task=4    # Número de núcleos por tarea
#SBATCH --time=02:00:00      # Tiempo máximo de ejecución (ajústalo según lo que necesites)
#SBATCH --partition=default  # Nombre de la partición (si no sabes, pregúntalo)

module load python-full      # Cargar el módulo de Python en el clúster
srun python 1_BETA.py -pr IMERG -tr 3h -ys 2002 -ye 2012  # Ejecuta tu script con los parámetros

