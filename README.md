# TnT
[TnT](https://openreview.net/pdf?id=qmsX2R19p9) is a table-language model framework that features multimodal table representations to empower LLMs to effectively and efficiently abstract structure-enriched semantics from tabular data.

---

## 📦 Installation

Create the environment with the required dependencies:

```bash
conda env create -f environment.yml
```

---

## **🚀** Training

We use [🤗Accelerate](https://huggingface.co/docs/accelerate/index) to train the model on multiple GPUs. An example configuration is provided under `accelerate_config/`. 

### **🔧 Before Training**

- Update the `config.py` file with the correct paths and hyperparameters.
- Some hyperparameters and data paths are also hardcoded in individual scripts — please double-check those.

### **🏋️ Training Scripts**

Scripts (for each base model) are organized in the `scripts/` directory. Key components include:

- **Table Encoder Pre-training**

  ```bash
  Table-Encoder/run.sh
  ```

- **Feature Alignment (multi-task)**

  ```bash
  scripts/align_multitask_codellama.sh
  scripts/align_multitask_llama3.sh
  scripts/align_multitask_mistral.sh
  ```

- **TnT Model SFT (Supervised Fine-Tuning)**

  ```bash
  scripts/sft_tnt_codellama.sh
  scripts/sft_tnt_llama3.sh
  scripts/sft_tnt_mistral.sh
  ```

- **Decoder-Only SFT**

  ```bash
  scripts/sft_decoder_codellama.sh
  scripts/sft_decoder_llama3.sh
  scripts/sft_decoder_mistral.sh
  ```

---

## **📊 Evaluation**

Evaluation scripts are located in the `eval_spider/` directory.

We use [test-suite-sql-eval](https://github.com/taoyds/test-suite-sql-eval) to assess SQL generation quality.

### **🧪 Setup**

1. Clone the test-suite-sql-eval repository.
2. Copy `eval_spider/eval_dict.py` into that repo.
3. Update the `EVALUATION_FILE` parameter in `config.py`.

### **🧾 Evaluation Scripts**

- **Decoder-only model evaluation**

  ```bash
  python eval_spider/gen_base.py             # For semantic datasets
  python eval_spider/gen_base_nonsem.py      # For non-semantic datasets
  ```

- **TnT / SP model evaluation**

  ```bash
  python eval_spider/gen_tnt.py              # For semantic datasets
  python eval_spider/gen_tnt_nonsem.py       # For non-semantic datasets
  ```

---

## **📁 Data Preparation**

### **🏋️ Training Data Format**

Training data should be a JSON list with the following structure:

```json
[
    {
        "instruction": "Given the following database schema:\ntable Faculty_Participates_in, columns = [Faculty_Participates_in.actid(<insert_embs>|int64), Faculty_Participates_in.FacID(<insert_embs>|int64)]\ntable Participates_in, columns = [Participates_in.stuid(<insert_embs>|int64), Participates_in.actid(<insert_embs>|int64)]\ntable Faculty, columns = [Faculty.Sex(<insert_embs>|object), Faculty.Lname(<insert_embs>|object), Faculty.FacID(<insert_embs>|int64|primary key), Faculty.Fname(<insert_embs>|object), Faculty.Rank(<insert_embs>|object), Faculty.Room(<insert_embs>|object), Faculty.Phone(<insert_embs>|int64), Faculty.Building(<insert_embs>|object)]\ntable Student, columns = [Student.Age(<insert_embs>|int64), Student.LName(<insert_embs>|object), Student.StuID(<insert_embs>|int64|primary key), Student.Major(<insert_embs>|int64), Student.Sex(<insert_embs>|object), Student.city_code(<insert_embs>|object), Student.Advisor(<insert_embs>|int64), Student.Fname(<insert_embs>|object)]\ntable Activity, columns = [Activity.actid(<insert_embs>|int64|primary key), Activity.activity_name(<insert_embs>|object)]\nForeign keys: ['Participates_in.actid=Activity.actid', 'Participates_in.stuid=Student.StuID']\n\nAnswer the following question with the corresponding sqlite SQL query only and with no explanation.\nQuestion: How many faculty members do we have for each faculty rank?",
        "answer": "SELECT rank ,  count(*) FROM Faculty GROUP BY rank",
        "path_csvs": [
            "csv_dir/table1.csv",
            "csv_dir/table2.csv",
            "... more tables ..."
        ]
  },
]
```

### **📑 Evaluation Data Format**

Evaluation data uses the same format as training data, with an added `db_id` field:

```json
[
    {
        "instruction": "Given the following database schema:\ntable stadium, columns = [stadium.Stadium_ID(<insert_embs>|int64|primary key), stadium.Location(<insert_embs>|object), stadium.Name(<insert_embs>|object), stadium.Capacity(<insert_embs>|int64), stadium.Highest(<insert_embs>|int64), stadium.Lowest(<insert_embs>|int64), stadium.Average(<insert_embs>|int64)]\ntable singer, columns = [singer.Singer_ID(<insert_embs>|int64|primary key), singer.Name(<insert_embs>|object), singer.Country(<insert_embs>|object), singer.Song_Name(<insert_embs>|object), singer.Song_release_year(<insert_embs>|int64), singer.Age(<insert_embs>|int64), singer.Is_male(<insert_embs>|object)]\ntable concert, columns = [concert.concert_ID(<insert_embs>|int64|primary key), concert.concert_Name(<insert_embs>|object), concert.Theme(<insert_embs>|object), concert.Stadium_ID(<insert_embs>|int64), concert.Year(<insert_embs>|int64)]\ntable singer_in_concert, columns = [singer_in_concert.concert_ID(<insert_embs>|int64), singer_in_concert.Singer_ID(<insert_embs>|int64)]\nForeign keys: ['concert.Stadium_ID=stadium.Stadium_ID', 'singer_in_concert.Singer_ID=singer.Singer_ID', 'singer_in_concert.concert_ID=concert.concert_ID']\n\nAnswer the following question with the corresponding sqlite SQL query only and with no explanation.\nQuestion: Show name, country, age for all singers ordered by age from the oldest to the youngest.",
        "answer": "SELECT name ,  country ,  age FROM singer ORDER BY age DESC",
        "path_csvs": [
            "csv_dir/table1.csv",
            "csv_dir/table2.csv",
            "... more tables ..."
        ],
        "db_id": "concert_singer"
    },
]
```