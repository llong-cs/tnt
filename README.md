# TNT
### install
* Install the requirements using `conda env create -f environment.yml`

### train
* We use [accelerate](https://huggingface.co/docs/accelerate/index) to train the model on multiple GPUs. Example config file is provided in `accelerate_config/` folder. Training scripts are provided in `scripts/` folder.
* Before running the training script, make sure to update the `config.py` file with the correct paths and hyperparameters. The scripts itself also have some hyperparameters and paths that should be updated.
* Scripts for encoder pre-training is `Table-Encoder/run.sh`
* Scripts for feature alignment are `align_multitask_(codellama / llama3 / mistral).sh`
* Scripts for SFT on TNT are `sft_tnt_(codellama / llama3 / mistral)`.sh
* Scripts for decoder-only SFT are `sft_decoder_(codellama / llama3 / mistral).sh`

### evaluation
* Evaluation scripts are provided in `eval_spider/` folder.
* We use [test-suite-sql-eval](https://github.com/taoyds/test-suite-sql-eval) to evaluate the generated SQL queries. Before running the evaluation script, make sure to clone the repository, put `eval_spider/eval_dict.py` in the folder, and update `EVALUATIION_FILE` parameter in config.py
* You can use `gen_base.py` or `gen_base_nonsem.py` to evaluate performance of any decoder-only model on semantic or non-semantic dataset
* You can use `gen_tnt.py` or `gen_tnt_nonsem.py` to evaluate performance of our tnt model or sp model on semantic or non-semantic dataset

### training data
You should prepare training data in a json file following this format:
```json
[
    {
        "instruction": "Given the following database schema:\ntable Faculty_Participates_in, columns = [Faculty_Participates_in.actid(<insert_embs>|int64), Faculty_Participates_in.FacID(<insert_embs>|int64)]\ntable Participates_in, columns = [Participates_in.stuid(<insert_embs>|int64), Participates_in.actid(<insert_embs>|int64)]\ntable Faculty, columns = [Faculty.Sex(<insert_embs>|object), Faculty.Lname(<insert_embs>|object), Faculty.FacID(<insert_embs>|int64|primary key), Faculty.Fname(<insert_embs>|object), Faculty.Rank(<insert_embs>|object), Faculty.Room(<insert_embs>|object), Faculty.Phone(<insert_embs>|int64), Faculty.Building(<insert_embs>|object)]\ntable Student, columns = [Student.Age(<insert_embs>|int64), Student.LName(<insert_embs>|object), Student.StuID(<insert_embs>|int64|primary key), Student.Major(<insert_embs>|int64), Student.Sex(<insert_embs>|object), Student.city_code(<insert_embs>|object), Student.Advisor(<insert_embs>|int64), Student.Fname(<insert_embs>|object)]\ntable Activity, columns = [Activity.actid(<insert_embs>|int64|primary key), Activity.activity_name(<insert_embs>|object)]\nForeign keys: ['Participates_in.actid=Activity.actid', 'Participates_in.stuid=Student.StuID']\n\nAnswer the following question with the corresponding sqlite SQL query only and with no explanation.\nQuestion: How many faculty members do we have for each faculty rank?",
        "answer": "SELECT rank ,  count(*) FROM Faculty GROUP BY rank",
        "path_csvs": [
            "csv_dir/csv_file_path",
            "csv_dir/csv_file_path",
            "csv_dir/csv_file_path",
            "csv_dir/csv_file_path",
            "csv_dir/csv_file_path"
        ]
  },
]
```

### evaluating data
You should prepare data for evaluation in a json file following this format:
```json
[
    {
        "instruction": "Given the following database schema:\ntable stadium, columns = [stadium.Stadium_ID(<insert_embs>|int64|primary key), stadium.Location(<insert_embs>|object), stadium.Name(<insert_embs>|object), stadium.Capacity(<insert_embs>|int64), stadium.Highest(<insert_embs>|int64), stadium.Lowest(<insert_embs>|int64), stadium.Average(<insert_embs>|int64)]\ntable singer, columns = [singer.Singer_ID(<insert_embs>|int64|primary key), singer.Name(<insert_embs>|object), singer.Country(<insert_embs>|object), singer.Song_Name(<insert_embs>|object), singer.Song_release_year(<insert_embs>|int64), singer.Age(<insert_embs>|int64), singer.Is_male(<insert_embs>|object)]\ntable concert, columns = [concert.concert_ID(<insert_embs>|int64|primary key), concert.concert_Name(<insert_embs>|object), concert.Theme(<insert_embs>|object), concert.Stadium_ID(<insert_embs>|int64), concert.Year(<insert_embs>|int64)]\ntable singer_in_concert, columns = [singer_in_concert.concert_ID(<insert_embs>|int64), singer_in_concert.Singer_ID(<insert_embs>|int64)]\nForeign keys: ['concert.Stadium_ID=stadium.Stadium_ID', 'singer_in_concert.Singer_ID=singer.Singer_ID', 'singer_in_concert.concert_ID=concert.concert_ID']\n\nAnswer the following question with the corresponding sqlite SQL query only and with no explanation.\nQuestion: Show name, country, age for all singers ordered by age from the oldest to the youngest.",
        "answer": "SELECT name ,  country ,  age FROM singer ORDER BY age DESC",
        "path_csvs": [
            "csv_dir/csv_file_path",
            "csv_dir/csv_file_path",
            "csv_dir/csv_file_path",
            "csv_dir/csv_file_path"
        ],
        "db_id": "concert_singer"
    },
]
```