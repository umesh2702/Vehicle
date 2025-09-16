import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DTC_FILE = DATA_DIR / "dtc_dataset.csv"

class DTCLookup:
    def __init__(self, dtc_csv: Path = DTC_FILE):
        if not dtc_csv.exists():
            raise FileNotFoundError(f"DTC file not found: {dtc_csv}")
        self.df = pd.read_csv(dtc_csv)
        # Normalize column names if needed
        self.df.columns = [c.strip() for c in self.df.columns]
        # Expect a Code column - try possible names
        possible_code_cols = [c for c in self.df.columns if c.lower() in ('code','dtc','error code','error_code')]
        if not possible_code_cols:
            raise ValueError("dtc_dataset.csv must contain a 'Code' or 'DTC' column.")
        code_col = possible_code_cols[0]
        self.df['Code'] = self.df[code_col].astype(str).str.strip()
        self.df = self.df.set_index('Code')

    def lookup(self, code: str):
        if code is None:
            return None
        code = str(code).strip()
        if code in self.df.index:
            row = self.df.loc[code]
            # Try to find human-friendly columns
            def get_col(*cands):
                for c in cands:
                    if c in row:
                        return row[c]
                return ''
            meaning = get_col('Meaning','Meaning of code','Description')
            cause = get_col('Possible Causes','Possible Cause','Possible_Cause','Cause')
            fix = get_col('Fix Suggestion','Fix','Action','Recommended Action')
            urgency = get_col('Urgency','Priority')
            return {
                'code': code,
                'meaning': meaning,
                'possible_cause': cause,
                'fix_suggestion': fix,
                'urgency': urgency
            }
        return None

if __name__ == "__main__":
    l = DTCLookup()
    print(l.lookup('P0141'))
