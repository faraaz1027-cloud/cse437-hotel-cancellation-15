import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.development_eda import read_selected_rows, rates, weighted_rates


class DevelopmentEdaTests(unittest.TestCase):
    def test_selected_rows_preserve_alignment_and_ignore_heldout_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/"values.csv"
            source=pd.DataFrame({"value":[2,10,3,20,4],"label":[0,1,1,0,0]})
            mask=np.array([True,False,True,False,True])
            source.to_csv(p,index=False)
            selected=read_selected_rows(p,mask,chunksize=2)
            self.assertEqual(selected.value.tolist(),[2,3,4])
            source.loc[~mask,:]=999
            source.to_csv(p,index=False)
            pd.testing.assert_frame_equal(selected,read_selected_rows(p,mask,chunksize=3))
            with self.assertRaisesRegex(ValueError,"row count"):
                read_selected_rows(p,np.ones(6,dtype=bool))

    def test_rates_account_for_every_booking_and_outcome(self):
        frame=pd.DataFrame({"category":["a","a","b"],"is_canceled":[0,1,1]})
        table=rates(frame,"category").set_index("category")
        self.assertEqual(table.bookings.sum(),3)
        self.assertEqual(table.canceled.sum(),2)
        self.assertEqual(table.loc["a","cancellation_percent"],50)

    def test_equal_group_weight_retains_conflicting_labels(self):
        frame=pd.DataFrame({"category":["a"]*4,"duplicate_group_id":[1,1,1,2],
                            "is_canceled":[1,1,0,0]})
        row=weighted_rates(frame,"category").iloc[0]
        self.assertEqual(row.duplicate_groups,2)
        self.assertAlmostEqual(row.equal_group_cancellation_percent,100/3)
        frame.loc[1,"category"]="b"
        with self.assertRaisesRegex(ValueError,"constant"):
            weighted_rates(frame,"category")


if __name__=="__main__":
    unittest.main()
