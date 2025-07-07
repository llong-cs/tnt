#!/bin/bash

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <source_dir> <target_dir>"
    exit 1
fi

source_dir="$1"
target_dir="$2"

mkdir -p "$target_dir"

for dir in "$source_dir"/*; do
    if [ -d "$dir" ]; then 
        dir_name=$(basename "$dir")
        for file in "$dir"/*.csv; do
            if [ -f "$file" ]; then
                new_file_name="${dir_name}_$(basename "$file")"
                cp "$file" "$target_dir/$new_file_name"
                echo "Copied $file to $target_dir/$new_file_name"
            fi
        done
    fi
done