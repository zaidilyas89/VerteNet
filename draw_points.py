import cv2
import numpy as np

colors = [[0.76590096, 0.0266074, 0.9806378],
           [0.54197179, 0.81682527, 0.95081629],
           [0.0799733, 0.79737015, 0.15173816],
           [0.93240442, 0.8993321, 0.09901344],
           [0.73130136, 0.05366301, 0.98405681],
           [0.01664966, 0.16387004, 0.94158259],
           [0.54197179, 0.81682527, 0.45081629],
           # [0.92074915, 0.09919099 ,0.97590748],
           [0.83445145, 0.97921679, 0.12250426],
           [0.7300924, 0.23253621, 0.29764521],
           [0.3856775, 0.94859286, 0.9910683],  # 10
           [0.45762137, 0.03766411, 0.98755338],
           [0.99496697, 0.09113071, 0.83322314],
           [0.96478873, 0.0233309, 0.13149931],
           [0.33240442, 0.9993321 , 0.59901344],
            # [0.77690519,0.81783954,0.56220024],
           # [0.93240442, 0.8993321, 0.09901344],
           [0.95815068, 0.88436046, 0.55782268],
           [0.03728425, 0.0618827, 0.88641827],
           [0.05281129, 0.89572238, 0.08913828],

           ]



def draw_landmarks_regress_test(pts0, ori_image_regress, ori_image_points):
    ori_image_points_points_only = ori_image_points.copy()
    for i, pt in enumerate(pts0):
        # color = np.random.rand(3)
        # color = colors[i]
        color = colors[10]
        # print(i+1, color)
        color_255 = (255 * color[0], 255 * color[1], 255 * color[2])
        cv2.circle(ori_image_regress, (int(pt[0]), int(pt[1])), 6, color_255, -1, 1)
        # cv2.circle(ori_image, (int(pt[2]), int(pt[3])), 5, color_255, -1,1)
        # cv2.circle(ori_image, (int(pt[4]), int(pt[5])), 5, color_255, -1,1)
        # cv2.circle(ori_image, (int(pt[6]), int(pt[7])), 5, color_255, -1,1)
        # cv2.circle(ori_image, (int(pt[8]), int(pt[9])), 5, color_255, -1,1)
        cv2.arrowedLine(ori_image_regress, (int(pt[0]), int(pt[1])), (int(pt[2]), int(pt[3])), (255 * 1, 255 * 0, 255 * 0), 2, 1,
                        tipLength=0.2)
        cv2.arrowedLine(ori_image_regress, (int(pt[0]), int(pt[1])), (int(pt[4]), int(pt[5])), (255 * 1, 255 * 0, 255 * 0), 2, 1,
                        tipLength=0.2)
        cv2.arrowedLine(ori_image_regress, (int(pt[0]), int(pt[1])), (int(pt[6]), int(pt[7])), (255 * 1, 255 * 0, 255 * 0), 2, 1,
                        tipLength=0.2)
        cv2.arrowedLine(ori_image_regress, (int(pt[0]), int(pt[1])), (int(pt[8]), int(pt[9])), (255 * 1, 255 * 0, 255 * 0), 2, 1,
                        tipLength=0.2)
        # cv2.putText(ori_image_regress, '{}'.format(i + 1),
        #             (int(pt[4] + 10), int(pt[5] + 10)),
        #             cv2.FONT_HERSHEY_DUPLEX,
        #             1.2,
        #             color_255,  # (255,255,255),
        #             1,
        #             1)
        # cv2.circle(ori_image, (int(pt[0]), int(pt[1])), 6, (255,255,255), -1,1)
        cv2.circle(ori_image_points, (int(pt[2]), int(pt[3])), 5, (255 * 0, 255 * 0, 255 * 1), -1, 1)
        cv2.circle(ori_image_points, (int(pt[4]), int(pt[5])), 5, (255 * 0, 255 * 0, 255 * 1), -1, 1)
        cv2.circle(ori_image_points, (int(pt[6]), int(pt[7])), 5, (255 * 0, 255 * 0, 255 * 1), -1, 1)
        cv2.circle(ori_image_points, (int(pt[8]), int(pt[9])), 5, (255 * 0, 255 * 0, 255 * 1), -1, 1)
        
        cv2.circle(ori_image_points_points_only, (int(pt[2]), int(pt[3])), 5, (255 * 0, 255 * 0, 255 * 1), -1, 1)
        cv2.circle(ori_image_points_points_only, (int(pt[4]), int(pt[5])), 5, (255 * 0, 255 * 0, 255 * 1), -1, 1)
        cv2.circle(ori_image_points_points_only, (int(pt[6]), int(pt[7])), 5, (255 * 0, 255 * 0, 255 * 1), -1, 1)
        cv2.circle(ori_image_points_points_only, (int(pt[8]), int(pt[9])), 5, (255 * 0, 255 * 0, 255 * 1), -1, 1)
        
        #----------------------Guide Lines for AAC Scoring--------------------------------
        Mean_L1_Left  = np.array([(pts0[0][6] + pts0[1][2])/2,(pts0[0][7] + pts0[1][3])/2])
        Mean_L1_Right = np.array([(pts0[0][8] + pts0[1][4])/2,(pts0[0][9] + pts0[1][5])/2])
        
        Mean_L2_Left  = np.array([(pts0[1][6] + pts0[2][2])/2,(pts0[1][7] + pts0[2][3])/2])
        Mean_L2_Right = np.array([(pts0[1][8] + pts0[2][4])/2,(pts0[1][9] + pts0[2][5])/2])
        
        Mean_L3_Left  = np.array([(pts0[2][6] + pts0[3][2])/2,(pts0[2][7] + pts0[3][3])/2])
        Mean_L3_Right = np.array([(pts0[2][8] + pts0[3][4])/2,(pts0[2][9] + pts0[3][5])/2])
        
        Mean_L4_Left  = np.array([(pts0[3][6] + pts0[4][2])/2,(pts0[3][7] + pts0[4][3])/2])
        Mean_L4_Right = np.array([(pts0[3][8] + pts0[4][4])/2,(pts0[3][9] + pts0[4][5])/2]) 
        
        Mean_L5_Left  = np.array([(pts0[4][6] + pts0[5][2])/2,(pts0[4][7] + pts0[5][3])/2])
        Mean_L5_Right = np.array([(pts0[4][8] + pts0[5][4])/2,(pts0[4][9] + pts0[5][5])/2])
    
        L1_L2_Line = np.polyfit((Mean_L1_Left[0],Mean_L1_Right[0]),(Mean_L1_Left[1],Mean_L1_Right[1]),1)
        L2_L3_Line = np.polyfit((Mean_L2_Left[0],Mean_L2_Right[0]),(Mean_L2_Left[1],Mean_L2_Right[1]),1)
        L3_L4_Line = np.polyfit((Mean_L3_Left[0],Mean_L3_Right[0]),(Mean_L3_Left[1],Mean_L3_Right[1]),1)
        L4_L5_Line = np.polyfit((Mean_L4_Left[0],Mean_L4_Right[0]),(Mean_L4_Left[1],Mean_L4_Right[1]),1)
        L5_L6_Line = np.polyfit((Mean_L5_Left[0],Mean_L5_Right[0]),(Mean_L5_Left[1],Mean_L5_Right[1]),1)
        
        max_x = int(max(np.concatenate([pts0[:,0],pts0[:,1],pts0[:,3],pts0[:,5],pts0[:,7],pts0[:,9]])))
        
        p1 = np.poly1d(L1_L2_Line)
        p2 = np.poly1d(L2_L3_Line)
        p3 = np.poly1d(L3_L4_Line)
        p4 = np.poly1d(L4_L5_Line)
        p5 = np.poly1d(L5_L6_Line)
        
               
        cv2.line(ori_image_points, (int(Mean_L1_Left[0]),int(Mean_L1_Left[1])),(int(Mean_L1_Right[0]+150),int(p1(Mean_L1_Right[0]+150))),(255 * 1, 255 * 0, 255 * 0),2)
        cv2.line(ori_image_points, (int(Mean_L2_Left[0]),int(Mean_L2_Left[1])),(int(Mean_L2_Right[0]+150),int(p2(Mean_L2_Right[0]+150))),(255 * 1, 255 * 0, 255 * 0),2)
        cv2.line(ori_image_points, (int(Mean_L3_Left[0]),int(Mean_L3_Left[1])),(int(Mean_L3_Right[0]+150),int(p3(Mean_L3_Right[0]+150))),(255 * 1, 255 * 0, 255 * 0),2)
        cv2.line(ori_image_points, (int(Mean_L4_Left[0]),int(Mean_L4_Left[1])),(int(Mean_L4_Right[0]+150),int(p4(Mean_L4_Right[0]+150))),(255 * 1, 255 * 0, 255 * 0),2)
        cv2.line(ori_image_points, (int(Mean_L5_Left[0]),int(Mean_L5_Left[1])),(int(Mean_L5_Right[0]+150),int(p5(Mean_L5_Right[0]+150))),(255 * 1, 255 * 0, 255 * 0),2)
    return ori_image_regress, ori_image_points, ori_image_points_points_only



def draw_landmarks_pre_proc(out_image, pts):
    # for i in range(17):
    for i in range(6):
        pts_4 = pts[4 * i:4 * i + 4, :]
        color = colors[i]
        color_255 = (255 * color[0], 255 * color[1], 255 * color[2])
        cv2.circle(out_image, (int(pts_4[0, 0]), int(pts_4[0, 1])), 5, color_255, -1, 1)
        cv2.circle(out_image, (int(pts_4[1, 0]), int(pts_4[1, 1])), 5, color_255, -1, 1)
        cv2.circle(out_image, (int(pts_4[2, 0]), int(pts_4[2, 1])), 5, color_255, -1, 1)
        cv2.circle(out_image, (int(pts_4[3, 0]), int(pts_4[3, 1])), 5, color_255, -1, 1)
    return np.uint8(out_image)


def draw_regress_pre_proc(out_image, pts):
    for i in range(17):
        pts_4 = pts[4 * i:4 * i + 4, :]
        pt = np.mean(pts_4, axis=0)
        color = colors[i]
        color_255 = (255 * color[0], 255 * color[1], 255 * color[2])
        cv2.arrowedLine(out_image, (int(pt[0]), int(pt[1])), (int(pts_4[0, 0]), int(pts_4[0, 1])), color_255, 2, 1,
                        tipLength=0.2)
        cv2.arrowedLine(out_image, (int(pt[0]), int(pt[1])), (int(pts_4[1, 0]), int(pts_4[1, 1])), color_255, 2, 1,
                        tipLength=0.2)
        cv2.arrowedLine(out_image, (int(pt[0]), int(pt[1])), (int(pts_4[2, 0]), int(pts_4[2, 1])), color_255, 2, 1,
                        tipLength=0.2)
        cv2.arrowedLine(out_image, (int(pt[0]), int(pt[1])), (int(pts_4[3, 0]), int(pts_4[3, 1])), color_255, 2, 1,
                        tipLength=0.2)
        cv2.putText(out_image, '{}'.format(i + 1), (int(pts_4[1, 0] + 10), int(pts_4[1, 1] + 10)),
                    cv2.FONT_HERSHEY_DUPLEX, 1.2, color_255, 1, 1)
    return np.uint8(out_image)