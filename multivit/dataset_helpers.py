import os
import sys
import logging
from collections import defaultdict
import numpy as np
import pandas as pd
from PIL import Image


def get_filenames(root, path_filter=None):
    for r, d, ff in os.walk(root):
        for f in ff:
            if '.jpg' in f.lower() or '.jpeg' in f.lower():
                if path_filter is None:
                    yield r, f
                elif isinstance(path_filter, str):
                    if path_filter in r:
                        yield r, f
                elif isinstance(path_filter, (tuple, list)):
                    if any(p in r for p in path_filter):
                        yield r, f


def extract_filename_components(filename):
    elems = list()
    elem = list()
    for c in filename:
        if len(elems) == 0:
            if c.isdigit():
                elems.append(''.join(elem).strip())
                elem = [c]
            else:
                elem.append(c)
        elif len(elems) == 1:
            if not c.isdigit():
                elems.append(''.join(elem).strip())
                elem = [c]
            else:
                elem.append(c)
        else:
            if c == '.':
                elems.append(''.join(elem).strip())
                break
            else:
                elem.append(c)

    return elems[1], elems[2]


def transform_with_processor(processor):
    def __inner(batch):
        input_images = [Image.open(img) for img in batch['image_name']]
        images = processor(input_images, return_tensors='pt')
        rbatch = dict(batch)
        rbatch['pixel_values'] = images['pixel_values']
        return rbatch
    return __inner


def get_image_entries_containing_poses(base_path, dataset_path, subfolder, required_poses=None, verbose=0):
    entries = list()
    poses = defaultdict(list)
    # pass 1: get all images of the appropriate poses
    for r, f in get_filenames(os.path.join(base_path, dataset_path), subfolder):
        _id, pose = extract_filename_components(f)
        if required_poses is None or pose in required_poses:
            rel = os.path.relpath(r, base_path)
            entries.append((os.path.join(rel, f), _id, pose))
            poses[id].append(pose)
    entries = sorted(entries)
    print("entries found:", len(entries))

    # pass 2: keep only patients with all the required poses if required_poses is provided
    if required_poses is not None:
        entries = [e for e in entries if len(poses[e[1]]) == len(required_poses)]
    poses = {k: sorted(v) for k, v in poses.items()}

    if verbose > 0:
        print("entries:", len(entries))
    if verbose > 1:
        for se in entries:
            print(se, poses[se[1]])

    return entries

def generate_integer_severity_categories(table):

    same_as_last = False
    ids = table['id']
    pasi_0 = table['pasi_0']
    pasi_1 = table['pasi_1']
    category = table['physician_ga']

    dest_categories = list()
    category_lookups = {
        "absent": 0,
        "0 - clear": 0,
        "1 - almost clear": 1,
        "2 - mild": 2,
        "3 - moderate": 3,
        "4 - moderate/severe": 4,
        "5 - severe": 5
    }

    for i in range(len(table.index)):
        dest_categories.append(category_lookups[category[i].lower()])

    return dest_categories


def convert_entries_to_pandas(entries):
    dict_of_entries = defaultdict(list)
    for e in entries:
        dict_of_entries['id'].append(int(e[1]))
        dict_of_entries['image_name'].append(e[0])
        dict_of_entries['pose'].append(e[2])

    df = pd.DataFrame(dict_of_entries)
    return df


def load_psoriasis_score_xlsx(path):
    edf = pd.read_excel(path)
    edf = edf[edf['Trunk_Induration'].notnull()]
    edf = edf.rename(
        columns={
            'ID': 'patient_id',
            'Trunk_Erythema': 'tr_erythema_0', 'Trunk_Induration': 'tr_induration_0', 'Trunk_Scaling': 'tr_scaling_0',
            'Trunk_Sum': 'tr_sum_0', 'Trunk_Area': 'tr_area_0', 'Trunk_SA': 'tr_sa_0', 'Trunk_T': 'tr_total_0',
            'UL_Erythema': 'ul_erythema_0', 'UL_induration': 'ul_induration_0', 'UL_Scaling': 'ul_scaling_0',
            'UL_Sum': 'ul_sum_0', 'UL_Area': 'ul_area_0', 'UL_SA': 'ul_sa_0', 'UL_T': 'ul_total_0',
            'LL_Erythema': 'll_erythema_0', 'LL_induration': 'll_induration_0', 'LL_Scaling': 'll_scaling_0',
            'LL_Sum': 'll_sum_0', 'LL_Area': 'll_area_0', 'LL_SA': 'll_sa_0', 'LL_T': 'll_total_0',
            'TT': 'total_0',
            'Trunk_Erythema.1': 'tr_erythema_1', 'Trunk_Induration.1': 'tr_induration_1', 'Trunk_Scaling.1': 'tr_scaling_1',
            'Trunk_Sum.1': 'tr_sum_1', 'Trunk_Area.1': 'tr_area_1', 'Trunk_SA.1': 'tr_sa_1', 'Trunk_T.1': 'tr_total_1',
            'UL_Erythema.1': 'ul_erythema_1', 'UL_induration.1': 'ul_induration_1', 'UL_Scaling.1': 'ul_scaling_1',
            'UL_Sum.1': 'ul_sum_1', 'UL_Area.1': 'ul_area_1', 'UL_SA.1': 'ul_sa_1', 'UL_T.1': 'ul_total_1',
            'LL_Erythema.1': 'll_erythema_1', 'LL_induration.1': 'll_induration_1', 'LL_Scaling.1': 'll_scaling_1',
            'LL_Sum.1': 'll_sum_1', 'LL_Area.1': 'll_area_1', 'LL_SA.1': 'll_sa_1', 'LL_T.1': 'll_total_1',
            'TT.1': 'total_1'})
    edf = edf.drop(columns=['Unnamed: 23'])
    edf.insert(0, 'id', [int(x[7:]) for x in edf['patient_id']])
    edf = edf.dropna()
    for c in ('id',):
        edf[c] = edf[c].astype('int32')

    for c in ('tr_erythema_0', 'tr_induration_0', 'tr_scaling_0', 'tr_sum_0', 'tr_area_0', 'tr_sa_0', 'tr_total_0',
              'ul_erythema_0', 'ul_induration_0', 'ul_scaling_0', 'ul_sum_0', 'ul_area_0', 'ul_sa_0', 'ul_total_0',
              'll_erythema_0', 'll_induration_0', 'll_scaling_0', 'll_sum_0', 'll_area_0', 'll_sa_0', 'll_total_0', 'total_0',
              'tr_erythema_1', 'tr_induration_1', 'tr_scaling_1', 'tr_sum_1', 'tr_area_1', 'tr_sa_1', 'tr_total_1',
              'ul_erythema_1', 'ul_induration_1', 'ul_scaling_1', 'ul_sum_1', 'ul_area_1', 'ul_sa_1', 'ul_total_1',
              'll_erythema_1', 'll_induration_1', 'll_scaling_1', 'll_sum_1', 'll_area_1', 'll_sa_1', 'll_total_1', 'total_1',):
        edf[c] = edf[c].astype('float32')
    return edf


def load_psoriasis_characteristics_xlsx(path, sheet):
    edf = pd.read_excel(path, sheet)
    stripped_names = {k: k.strip() for k in edf.columns}
    edf = edf.rename(columns=stripped_names)

    edf = edf.rename(columns={
        'Coded Index': 'patient_id', 'Notes': 'notes', 'Photo modified?': 'photo_modified', 'Self-taken': 'self_taken',
        'Sex': 'sex', 'Ethnicity (use categories in PsoProtectMe)': 'ethnicity', 'Skin type': 'skin_type', 'Height (cm)': 'height_cm', 'Weight (kg)': 'weight_kg',
        'BMI': 'bmi', 'Age onset of psoriasis': 'age_onset', 'Patient GA (DrDoctor)': 'drdoctor_ga',
        'First PASI': 'pasi_0', 'Second PASI': 'pasi_1', 'Physician Global Assessment': 'physician_ga', 'BSA (%)': 'bsa_pct',
    })
    edf['notes'] = edf['notes'].fillna('')

    for c in ('patient_id', 'notes', 'photo_modified', 'self_taken', 'sex', 'ethnicity', 'skin_type', 'drdoctor_ga', 'physician_ga', 'age_onset'):
        edf[c] = edf[c].str.strip()
    for c in ('height_cm', 'weight_kg', 'bmi', 'pasi_0', 'pasi_1', 'bsa_pct'):
        edf[c] = edf[c].replace('Absent', np.nan)
        edf[c] = edf[c].astype('float32')
    return edf


def use_consistent_clinical_images_v0_1(dataframe):
    if logging in sys.modules:
        logger = logging.getLogger("use_consistent_clinical_images_v0_1")
    filtered_df = dataframe[dataframe['pose'].isin(('AP', 'PA', 'LL', 'RL'))]
    filtered_df = filtered_df[filtered_df.groupby('id')['id'].transform('size') == 4]
    if logging in sys.modules:
        logger.info(f"filtering dataset: {len(dataframe.index)} -> {len(filtered_df.index)} entries")
    return filtered_df


def use_all_clinical_images_v0_1(dataframe):
    if logging in sys.modules:
        logger = logging.getLogger("use_all_clinical_images_v0_1")
    filtered_df = dataframe[~dataframe['pose'].isin(('SA', 'SL', 'ST', 'SW'))]
    if logging in sys.modules:
        logger.info(f"filtering dataset: {len(dataframe.index)} -> {len(filtered_df.index)} entries")
    return filtered_df


def use_all_selfie_images_v0_1(dataframe):
    if logging in sys.modules:
        logger = logging.getLogger("use_all_selfie_image_sv0_1")
    filter_df = dataframe[dataframe['pose'].isin(('SA', 'SL', 'ST', 'SW'))]
    if logging in sys.modules:
        logger.info(f"filtering dataset: {len(dataframe.index)} -> {len(filtered_df.index)} entries")
    return filtered_df


def generate_folds_v0_1(dataframe, name='fold'):
    logger = logging.getLogger("generate_folds_v0_1")
    patient_ids = dataframe['id'].unique()
    logger.info(patient_ids)
    count = len(patient_ids)
    fold_size = count // 5
    folds = [fold_size] * 5
    for i in range(count % 5):
        folds[i] += 1
    logger.info(f"patient id count: {count}, base fold_size: {fold_size}, folds: {folds}")
    fold_entries_for_id = np.asarray([i_f for i_f in range(len(folds)) for _ in range(folds[i_f])], dtype=int)
    logger.info(fold_entries_for_id)
    fold_df = pd.DataFrame({'id': patient_ids, name: fold_entries_for_id})
    dataframe = dataframe.merge(fold_df, how='inner', left_on='id', right_on='id')

    return dataframe


def generate_categories_from_pasi(table, fig=None, ax=None):

    same_as_last = False
    ids = table['id']
    pasi_0 = table['pasi_0']
    pasi_1 = table['pasi_1']
    category = table['physician_category']

    dest_ids = list()
    dest_categories = list()
    dest_pasi_0s = list()

    last_id = None
    print(len(table.index))
    for i in range(len(table.index)):
        if ids[i] != last_id:
            # print(ids[i], pasi_0[i], pasi_1[i], category[i])
            last_id = ids[i]
            dest_ids.append(ids[i])
            dest_categories.append(category[i])
            dest_pasi_0s.append(pasi_0[i])


    if ax is not None:
        ax.scatter(dest_categories, dest_pasi_0s)

    return pd.DataFrame({'id': dest_ids, 'pasi_0': dest_pasi_0s, 'pasi_category': dest_categories})


def get_fields_per_patient(table, key, fields):

    key_field = table[key]
    patient_fields = [table[f] for f in fields]

    dest_key_field = list()
    dest_patient_fields = list()

    last_k = None
    for i in range(len(table.index)):
        if key_field[i] != last_k:
            dest_key_field.append(key_field[i])
            last_k = key_field[i]
            dest_patient_fields.append([patient_fields[f][i] for f in range(len(fields))])

    df = pd.DataFrame({key: dest_key_field})
    for i_f, f in enumerate(fields):
        df[f] = [d[i_f] for d in dest_patient_fields]

    return df


def get_category_ranges(values, categories):

    cats = sorted(list(set(categories)))
    means = []
    stds = []

    for cat in cats:
        cat_values = [v for v, c in zip(values, categories) if c == cat]
        means.append(np.mean(cat_values))
        stds.append(np.std(cat_values))

    # Function to find intersection points of two Gaussian distributions
    def find_intersection(mean1, std1, mean2, std2):
        a = 1/(2*std1**2) - 1/(2*std2**2)
        b = mean2/(std2**2) - mean1/(std1**2)
        c = mean1**2 /(2*std1**2) - mean2**2 / (2*std2**2) - np.log(std2/std1)
        return np.roots([a, b, c])

    intersections = list()
    for i in range(len(means)-1):
        intersections.append(find_intersection(means[i], stds[i], means[i+1], stds[i+1]))

    intersections = [max(i[0], i[1]) for i in intersections]

    print("Intersection points:", intersections)

    if fig is not None:
        fig, ax = plt.subplots(1, 1, figsize=(12, 4))
        x_axis = np.linspace(0, 60, 241)
        for m, s in zip(means, stds):
            ax.plot(x_axis, norm.pdf(x_axis, m, s))

    return [0] + intersections


def scores_to_categories(scores, category_thresholds):

    def score_to_category(score):
        for i in range(1, len(category_thresholds)):
            if score < category_thresholds[i]:
                return i
        return i+1

    if isinstance(scores, np.ndarray):
        categories = np.vectorize(score_to_category)(scores)
    else:
        categories = list()

        for s in scores:
            categories.append(score_to_category(s))

    return categories
