"""Modify the LTCs in the 2000 bus case to regulate from 0.9 to 1.1."""
from esa import SAW
import os

# Get full path to this directory.
THIS_DIR = os.path.dirname(os.path.abspath(__file__))

case = os.path.join(THIS_DIR, 'ACTIVSg2000_AUG-09-2018_Ride.PWB')
case_mod = os.path.join(THIS_DIR, 'ACTIVSg2000_AUG-09-2018_Ride_mod.PWB')

saw = SAW(FileName=case, early_bind=True)

kf = saw.get_key_field_list('branch')

fields = kf + ['XFRegMax', 'XFTapMax', 'LineXFType']

branch = saw.GetParametersMultipleElement(
    ObjectType='branch', ParamList=fields
)

ltc = branch.loc[branch['LineXFType'] == 'LTC', :].copy()

ltc['XFRegMax'] = 1.1
ltc['XFTapMax'] = 1.1

saw.change_and_confirm_params_multiple_element(
    ObjectType='branch', command_df=ltc)

saw.SaveCase(FileName=case_mod, FileType='PWB', Overwrite=True)

pass