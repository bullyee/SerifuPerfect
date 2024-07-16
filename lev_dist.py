import numpy as np


def str_similarity(str1, str2, insert_costs=None, delete_costs=None, replace_costs=None):
    """Calculate the similarity between two string, Based on levenshtein distance.
    Custom insert/delete/replace costs on certain characters are available.
        Args:
            str, img2: Two images with the same size
            bboxes ([cord1, cord2, cord3, cord4]): The focused(compared) part of the images.

        Returns:
            float: The difference ratio.
    """
    if insert_costs is None:
        insert_costs = {}
    if delete_costs is None:
        delete_costs = {}
    if replace_costs is None:
        replace_costs = {}
    default_edit_costs = [1, 1, 1]  # insert, delete, replace
    n, m = len(str1), len(str2)
    dp_table = np.zeros((n + 1, m + 1))

    for i in range(n + 1):
        dp_table[i][0] = i
    for j in range(m + 1):
        dp_table[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if str1[i - 1] == str2[j - 1]:
                dp_table[i][j] = dp_table[i - 1][j - 1]
            else:
                edit_costs = default_edit_costs
                if str1[i - 1] in insert_costs:
                    edit_costs[0] = insert_costs[str1[i - 1]]
                if str2[j - 1] in delete_costs:
                    edit_costs[1] = delete_costs[str2[j - 1]]
                if (str1[i - 1], str2[j - 1]) in replace_costs:
                    edit_costs[2] = replace_costs[(str1[i - 1], str2[j - 1])]
                elif (str2[j - 1], str1[i - 1]) in replace_costs:
                    edit_costs[2] = replace_costs[(str2[j - 1], str1[i - 1])]
                dp_table[i][j] = min(int(dp_table[i - 1][j]) + edit_costs[0], int(dp_table[i][j - 1]) + edit_costs[1],
                                     int(dp_table[i - 1][j - 1]) + edit_costs[2])
    return 1 - dp_table[n][m] / max(m, n)