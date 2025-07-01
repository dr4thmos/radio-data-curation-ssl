#!/bin/bash

# Salva come monitor_resources.sh
job_id=$1
log_file="memory_monitor_${job_id}.log"
interval=60  # Registra ogni 60 secondi

echo "Timestamp,RSS_Memory_GB,Virtual_Memory_GB,CPU_Usage_%" > $log_file

while true; do
    timestamp=$(date "+%Y-%m-%d %H:%M:%S")
    mem_stats=$(ps -u $USER -o rss,vsz,pcpu | grep -v RSS | sort -nr | head -n 1)
    rss_kb=$(echo $mem_stats | awk '{print $1}')
    vsz_kb=$(echo $mem_stats | awk '{print $2}')
    cpu_pct=$(echo $mem_stats | awk '{print $3}')
    
    rss_gb=$(echo "scale=2; $rss_kb/1024/1024" | bc)
    vsz_gb=$(echo "scale=2; $vsz_kb/1024/1024" | bc)
    
    echo "$timestamp,$rss_gb,$vsz_gb,$cpu_pct" >> $log_file
    sleep $interval
done