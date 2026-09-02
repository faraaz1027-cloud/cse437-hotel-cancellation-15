# Step 15 - remaining submission checks

Owner: **Sadat**, with **Faraaz reviewing the earlier stages and both reviewing
the complete report**. The report/PDF is assembled; the project is not yet
declared submission-ready. There is no Step 16.

## Completed and verified

- 10-page report/report.pdf and corresponding report/report.md, following the
  faculty's numbered sections; 168-word summary; course Section 05, Summer 2026.
- Frozen baseline/LR/RF test comparison, error examples, ROC/PR curves,
  confusion matrix, coefficient plot, limitations and AI-assistance disclosure.
- 58 unit tests; canonical format validation for all five notebooks.
- Python 3.12 environment with installed direct dependencies and a successful
  `pip check`. The Linux resolution is recorded in requirements-lock.txt.
- Original source DOCX, raw data and frozen Step 12/13/15 scientific evidence
  preserved. Generated report plots only read those saved results.

## 1. Obtain the repository and raw CSV

Clone the public repository, or use GitHub's **Code > Download ZIP**, extract
it, then open a terminal in the extracted repository folder. Do not work from
inside the ZIP. Keep the original CSE437 DOCX private and unchanged.

Place the supplied untouched hotel_bookings.csv at data/raw/hotel_bookings.csv.
If necessary, obtain it from the [documented dataset source](data/README.md).
The expected SHA-256 is:

`7c2ae42a7353905ea136e5c2287f17c92c5435826598bfbb8491c6f0c7b1fc06`

This 16.86 MB file must also be committed to the public repository under the
faculty's under-50-MB rule. Preserve the source/CC BY 4.0 attribution in
data/README.md. Its original acquisition date/version remain unknown; do not
replace those facts with the date you download another copy.

## 2. Prepare Python and run the local checks

Use Python 3.12. On Windows, from the repository folder:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m pip check
.venv\Scripts\python.exe -m unittest discover -s tests -q
.venv\Scripts\python.exe scripts/verify_notebooks.py
```

On macOS/Linux, substitute `python3.12 -m venv .venv` and
`.venv/bin/python` for the Windows Python executable. The lock file records the
resolved Linux environment; use the direct requirements file for other
platforms, then save the versions you actually used. Do not commit .venv/.

The runner creates a separate temporary repository copy and a new Jupyter
kernel for each notebook, then executes 01 through 05 in order. Its final output
must say `status: passed` with notebook_count 5 and zero errors. It prints the
location of verification.json and the executed notebook copies. Copy/save that
verification record and provide it for review. If it fails, keep the error and
do not mark the check complete. The source data/models/notebooks are checked
for changes and are not overwritten by the runner.

Fresh-kernel execution could not run on the assistant's host because kernel
startup encountered an operating-system permission restriction. The user
approved publishing the report with this check pending. Format validation and
unit tests are not substitutes for fresh-kernel execution. The runner itself
has not completed an end-to-end successful execution here.

Notebook 05 verifies saved test evidence; it does not retune models. Notebook
04 reproduces development tuning. Do not change model selection, features,
hyperparameters or threshold in response to held-out scores. Have the executed
notebooks and verification record reviewed before replacing published outputs.

## 3. Confirm genuine contributions

Each member must state what they personally implemented, checked or explained,
and include their own commit links where applicable. The report currently
records assigned scopes only, explicitly unconfirmed. AI-generated code is
disclosed; account identity is not evidence of personal authorship.

- Faraaz: review the earlier data/audit/preprocessing/EDA stages and Sections 1-3.
- Sadat: review feature engineering, modeling, tuning, evaluation and assembly.
- Both: review the final report, limits of the results, references and source terms.

Only after confirmation, update report Section 9 with truthful statements and
rebuild the PDF. From the repository root: `python scripts/build_report_pdf.py`.
It uses the committed Markdown and plots, verifies the summary length and
requires exactly 10 pages. Review every rendered page after any report edit.

## 4. Final submission

Confirm the raw CSV, five executed notebooks, code, requirements, all report
figures, saved final model, report.md and report.pdf are accessible publicly.
Check that every member has committed their actual work from their own account.
Remove pending claims only when the corresponding checks genuinely pass.

Submit **one public repository link** through the faculty's designated channel:

https://github.com/faraaz1027-cloud/cse437-hotel-cancellation-15

No submission is made automatically by generating or committing this report.
