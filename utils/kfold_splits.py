
def get_nested_kfold_splits(cases):

    splits = []
    for test_idx in range(len(cases)):
        for valid_idx in range(len(cases)):
            if test_idx != valid_idx:
                # Separate out train, valid, and test sets
                test_case = [cases[test_idx]]
                valid_case = [cases[valid_idx]]
                train_cases = [case for i, case in enumerate(cases) if i != test_idx and i != valid_idx]
                splits.append((train_cases, valid_case, test_case))

    return splits


def get_nested_kfold_sklearn(cases):

    from sklearn.model_selection import KFold

    # Number of folds for outer and inner splits
    outer_folds = len(cases)
    inner_folds = len(cases) - 1  # Inner splits will leave one out as test in the outer loop

    # Outer KFold for test sets
    outer_kf = KFold(n_splits=outer_folds)

    # Store the splits in a list
    nested_splits = []

    for train_valid_index, test_index in outer_kf.split(cases):
        # Assign the outer test set
        test_set = [cases[i] for i in test_index]
        train_valid_set = [cases[i] for i in train_valid_index]

        # Inner KFold for train and validation sets
        inner_kf = KFold(n_splits=inner_folds)
        for train_index, valid_index in inner_kf.split(train_valid_set):
            train_set = [train_valid_set[i] for i in train_index]
            valid_set = [train_valid_set[i] for i in valid_index]
            
            # Append each (train, validation, test) split to nested_splits
            nested_splits.append((train_set, valid_set, test_set))

    return nested_splits


if __name__ == '__main__':

    cases = ['s1', 's2', 's3', 's4']
    nested_splits = get_nested_kfold_splits(cases)
    nested_splits_skl = get_nested_kfold_sklearn(cases)

    for i, split in enumerate(nested_splits):
        train_set, test_set, valid_set = split
        print(f"Split {i+1}: Train: {train_set}, Validation: {valid_set}, Test: {test_set}")

    for i, split in enumerate(nested_splits_skl):
        train_set, test_set, valid_set = split
        print(f"Split {i+1}: Train: {train_set}, Validation: {valid_set}, Test: {test_set}")
