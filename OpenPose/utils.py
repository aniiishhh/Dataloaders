import cv2
import numpy as np


def getKeypoints(probMap, threshold=0.1):
    mapSmooth = cv2.GaussianBlur(probMap, (3, 3), 0, 0)

    mapMask = np.uint8(mapSmooth > threshold)
    keypoints = []

    contours, _ = cv2.findContours(mapMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        blobMask = np.zeros(mapMask.shape)
        blobMask = cv2.fillConvexPoly(blobMask, cnt, 1)
        maskedProbMap = mapSmooth * blobMask
        _, maxVal, _, maxLoc = cv2.minMaxLoc(maskedProbMap)
        keypoints.append(maxLoc + (probMap[maxLoc[1], maxLoc[0]],))

    return keypoints


def getValidPairs(output, mapIdx, POSE_PAIRS, image, detected_keypoints):
    valid_pairs = []
    invalid_pairs = []
    n_interp_samples = 10
    paf_score_th = 0.1
    conf_th = 0.7

    frameWidth = image.shape[1]
    frameHeight = image.shape[0]

    for k in range(len(mapIdx)):
        pafA = output[0, mapIdx[k][0], :, :]
        pafB = output[0, mapIdx[k][1], :, :]
        pafA = cv2.resize(pafA, (frameWidth, frameHeight))
        pafB = cv2.resize(pafB, (frameWidth, frameHeight))

        candA = detected_keypoints[POSE_PAIRS[k][0]]
        candB = detected_keypoints[POSE_PAIRS[k][1]]
        nA = len(candA)
        nB = len(candB)

        if nA != 0 and nB != 0:
            valid_pair = np.zeros((0, 3))
            for i in range(nA):
                max_j = -1
                maxScore = -1
                found = 0
                for j in range(nB):
                    d_ij = np.subtract(candB[j][:2], candA[i][:2])
                    norm = np.linalg.norm(d_ij)
                    if norm:
                        d_ij = d_ij / norm
                    else:
                        continue

                    interp_coord = list(
                        zip(
                            np.linspace(candA[i][0], candB[j][0], num=n_interp_samples),
                            np.linspace(candA[i][1], candB[j][1], num=n_interp_samples),
                        )
                    )

                    paf_interp = []
                    for k in range(len(interp_coord)):
                        paf_interp.append(
                            [
                                pafA[
                                    int(round(interp_coord[k][1])),
                                    int(round(interp_coord[k][0])),
                                ],
                                pafB[
                                    int(round(interp_coord[k][1])),
                                    int(round(interp_coord[k][0])),
                                ],
                            ]
                        )

                    paf_scores = np.dot(paf_interp, d_ij)
                    avg_paf_score = sum(paf_scores) / len(paf_scores)

                    if (
                        len(np.where(paf_scores > paf_score_th)[0]) / n_interp_samples
                    ) > conf_th:
                        if avg_paf_score > maxScore:
                            max_j = j
                            maxScore = avg_paf_score
                            found = 1

                if found:
                    valid_pair = np.append(
                        valid_pair, [[candA[i][3], candB[max_j][3], maxScore]], axis=0
                    )

            valid_pairs.append(valid_pair)
        else:
            print("No Connection : k = {}".format(k))
            invalid_pairs.append(k)
            valid_pairs.append([])
    return valid_pairs, invalid_pairs


def getPersonwiseKeypoints(
    valid_pairs, invalid_pairs, mapIdx, POSE_PAIRS, keypoints_list
):
    personwiseKeypoints = -1 * np.ones((0, 19))

    for k in range(len(mapIdx)):
        if k not in invalid_pairs:
            partAs = valid_pairs[k][:, 0]
            partBs = valid_pairs[k][:, 1]
            indexA, indexB = np.array(POSE_PAIRS[k])

            for i in range(len(valid_pairs[k])):
                found = 0
                person_idx = -1
                for j in range(len(personwiseKeypoints)):
                    if personwiseKeypoints[j][indexA] == partAs[i]:
                        person_idx = j
                        found = 1
                        break

                if found:
                    personwiseKeypoints[person_idx][indexB] = partBs[i]
                    personwiseKeypoints[person_idx][-1] += (
                        keypoints_list[partBs[i].astype(int), 2] + valid_pairs[k][i][2]
                    )

                elif not found and k < 17:
                    row = -1 * np.ones(19)
                    row[indexA] = partAs[i]
                    row[indexB] = partBs[i]

                    row[-1] = (
                        sum(keypoints_list[valid_pairs[k][i, :2].astype(int), 2])
                        + valid_pairs[k][i][2]
                    )
                    personwiseKeypoints = np.vstack([personwiseKeypoints, row])
    return personwiseKeypoints
