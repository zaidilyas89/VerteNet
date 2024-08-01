import torch
from sys import exit
import numpy as np
from models import guidenet_pp
import cv2
import decoder
import os
from dataset import BaseDataset
import draw_points, draw_points_detailed
import scipy.io
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import torchprofile


import torch
import torch.utils.benchmark as benchmark

# Define your model
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)   




def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def create_folders_if_not_exist(main_folder, subfolders):
    try:
        os.makedirs(main_folder)
        print(f"Main folder created: {main_folder}")
    except FileExistsError:
        print(f"Main folder already exists: {main_folder}")

    for subfolder in subfolders:
        subfolder_path = os.path.join(main_folder, subfolder)
        try:
            os.makedirs(subfolder_path)
            print(f"Subfolder created: {subfolder_path}")
        except FileExistsError:
            print(f"Subfolder already exists: {subfolder_path}")

def apply_mask(image, mask, alpha=0.5):
    """Apply the given mask to the image.
    """
    color = np.random.rand(3)
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image

class Network(object):
    def __init__(self, args):
        torch.manual_seed(317)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cuda")
        heads = {'hm': args.num_classes,
                 'reg': 2*args.num_classes,
                 'wh': 2*4,}

        self.model = guidenet_pp.guidenet_pp(heads=heads,
                                         pretrained=True,
                                         down_ratio=args.down_ratio,
                                         final_kernel=1,
                                         head_conv=256)
        
        
        self.num_classes = args.num_classes
        self.decoder = decoder.DecDecoder(K=args.K, conf_thresh=args.conf_thresh)
        self.dataset = {'spinal': BaseDataset}

    def load_model(self, model, resume):
        checkpoint = torch.load(resume, map_location=lambda storage, loc: storage)
        print('loaded weights from {}, epoch {}'.format(resume, checkpoint['epoch']))
        state_dict_ = checkpoint['state_dict']
        model.load_state_dict(state_dict_, strict=False)
        return model

    def map_mask_to_image(self, mask, img, color=None):
        if color is None:
            color = np.random.rand(3)
        mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
        mskd = img * mask
        clmsk = np.ones(mask.shape) * mask
        clmsk[:, :, 0] = clmsk[:, :, 0] * color[0] * 256
        clmsk[:, :, 1] = clmsk[:, :, 1] * color[1] * 256
        clmsk[:, :, 2] = clmsk[:, :, 2] * color[2] * 256
        img = img + 1. * clmsk - 1. * mskd
        return np.uint8(img)


    def test(self, args, save):
        save_path = 'weights_'+args.dataset
        self.model = self.load_model(self.model, os.path.join(save_path, args.resume))
        self.model = self.model.to(self.device)
        self.model.eval()

        dataset_module = self.dataset[args.dataset]
        dsets = dataset_module(data_dir=args.data_dir,
                               phase='test',
                               input_h=args.input_h,
                               input_w=args.input_w,
                               down_ratio=args.down_ratio)

        data_loader = torch.utils.data.DataLoader(dsets,
                                                  batch_size=1,
                                                  shuffle=False,
                                                  num_workers=0,
                                                  pin_memory=True)


        for cnt, data_dict in enumerate(data_loader):
            images = data_dict['images'][0]
            img_id = data_dict['img_id'][0]
            images = images.to('cuda')
            print('processing {}/{} image ... {}'.format(cnt, len(data_loader), img_id))
            with torch.no_grad():
                output = self.model(images)
                
                self.model.eval()  # Set the model to evaluation mode
                
                
                # ###############Inference Time code#######################
                # dummy_input = torch.randn(1, 3, 1024, 512).cuda()

                # # Calculate FLOPs using torchprofile
                # flops = torchprofile.profile_macs(self.model, dummy_input)
                
                # # Instantiate your model
                # # model = MyModel()
                # self.model.eval()  # Set the model to evaluation mode
                
                # # Generate some random input data
                # input_data = torch.randn(1, 3,1024,512).cuda()
                
                # # Run inference and measure the time
                # timer = benchmark.Timer(
                #     stmt="model(input_data)",
                #     globals={"model": self.model, "input_data": input_data}
                # )
                
                # # Print the inference time
                # print("Inference time:", timer.timeit(10))  # Adjust the number of iterations as needed
                # # print(f'Total FLOPs: {flops}')
                # ###############Inference Time code#######################
                
                hm = output['hm']
                wh = output['wh']
                reg = output['reg']

            torch.cuda.synchronize(self.device)
            pts2 = self.decoder.ctdet_decode(hm, wh, reg)   # 6, 11
            pts0 = pts2.copy()
            pts0[:,:10] *= args.down_ratio

            print('totol pts num is {}'.format(len(pts2)))

            ori_image = dsets.load_image(dsets.img_ids.index(img_id))
            ori_image_regress = cv2.resize(ori_image, (args.input_w, args.input_h))
            ori_image_points = ori_image_regress.copy()
            orig = ori_image_regress.copy()
            h,w,c = ori_image.shape
            pts0 = np.asarray(pts0, np.float32)
            # pts0[:,0::2] = pts0[:,0::2]/args.input_w*w
            # pts0[:,1::2] = pts0[:,1::2]/args.input_h*h
            sort_ind = np.argsort(pts0[:,1])
            pts0 = pts0[sort_ind]

    
            ori_image_regress, ori_image_points, ori_image_points_points_only, pts4 = draw_points_detailed.draw_landmarks_regress_test(pts0,
                                                                                           ori_image_regress,
                                                                                           ori_image_points)
            # plt.figure()
            # plt.imshow(ori_image_points, cmap='gray')
            # # plt.savefig(str(cnt))
            # plt.savefig(os.path.join('./temporary_outputs',img_id))
            # plt.close()
            SAVE_MORE_INFO = True
            if SAVE_MORE_INFO == True:
                
                # Example usage:
                main_folder = "detailed_outputs"
                subfolders = ["landmarks", "original_images", "images_with_landmarks_only", "images_with_landmarks_and_IVGs", "images_with_offset_vectors"]
                
                create_folders_if_not_exist(main_folder, subfolders)
                
                img1 = Image.fromarray(ori_image_regress)
                img2 = Image.fromarray(ori_image_points)
                img3 = Image.fromarray(ori_image_points_points_only)
                img4 = Image.fromarray(orig)
                
                df = pd.DataFrame(pts0, columns=['center_x','center_y','top_left_x','top_left_y','top_right_x','top_right_y','bottom_left_x','bottom_left_y','bottom_right_x','bottom_right_y','scale_factor'])
                df = df.set_index(pd.Index(['T12','L1','L2','L3','L4','L5']))
                
                img1.save(os.path.join(main_folder, subfolders[4],img_id.split('.')[0]+'.png'))
                img2.save(os.path.join(main_folder, subfolders[3],img_id.split('.')[0]+'.png'))
                img3.save(os.path.join(main_folder, subfolders[2],img_id.split('.')[0]+'.png'))
                img4.save(os.path.join(main_folder, subfolders[1],img_id.split('.')[0]+'.png'))
                df.to_csv(os.path.join(main_folder, subfolders[0],img_id.split('.')[0]+'.csv'))
            else:
                fig, axes = plt.subplots(1, 2)
                # Show the first image
                axes[0].imshow(orig)
                axes[0].axis('off')  # Turn off axis
                axes[0].set_title(img_id)
                
                # Show the second image
                axes[1].imshow(ori_image_points)
                axes[1].axis('off')  # Turn off axis
                axes[1].set_title('With_Landmarks')
                
                # Adjust layout
                plt.tight_layout()
                
                # Display the plot
                plt.show()
                plt.savefig(os.path.join('./temporary_outputs',img_id))
                plt.close()
            
            
            
            # cv2.imshow('ori_image_regress', ori_image_regress)
            # cv2.imshow('ori_image_points', ori_image_points)
            # k = cv2.waitKey(0) & 0xFF
            # cv2.destroyAllWindows()
            # matrix = {'pts0':pts0}
            # scipy.io.savemat(img_id+'.mat',matrix)
            # cv2.imwrite('./'+str(cnt)+'.png',ori_image_regress)
            # cv2.imwrite('./'+str(cnt)+str(cnt)+'.png',ori_image_points)
            # #path = 'D:/PhD Edith Cowan University/Online/Experimentation/Landmark Estimation/Images'
            # #cv2.imwrite(os.path.join(path,img_id +'.png'),ori_image_regress)
            # #cv2.imwrite('D:\PhD Edith Cowan University\Online\Experimentation\Landmark Estimation\''+str(cnt)+'.png',ori_image_points)
            
            # if k == ord('n'):
            #     l=1   
            #     # scipy.io.savemat(img_id+'.mat',matrix)
            #     # cv2.imwrite('D:\PhD Edith Cowan University\Online\Experimentation\Landmark Estimation\Generated Images'+str(cnt)+'.png',ori_image_regress)
            #     # cv2.imwrite('D:\PhD Edith Cowan University\Online\Experimentation\Landmark Estimation\Generated Corner Offsets'+str(cnt)+'.png',ori_image_points)
                
            # elif k == ord('e'):
            #     cv2.destroyAllWindows()
            #     exit()
