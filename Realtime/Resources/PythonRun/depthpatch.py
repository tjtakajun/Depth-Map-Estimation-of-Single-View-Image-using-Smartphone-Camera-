import argparse
from skimage import color
from skimage.io import imread, imsave
import time
import matplotlib.pyplot as plt
import numpy as np
from skimage.color import rgb2gray, rgb2lab
from skimage.filters import laplace
from scipy.ndimage.filters import convolve
import cv2


#from inpainter import Inpainter
#from Median import Median


def main():
    args = parse_args()

    print(args.model_image)
    model_image = imread(args.model_image)
    model_depth = rgb2gray(imread(args.model_depth))
    image = imread(args.image)
    patch_size = args.patch_size

    output_image = Inpainter(
        model_image,
        model_depth,
        image,
        patch_size,
        plot_progress=args.plot_progress
    ).inpaint()

    output_image = Median(output_image).median()

    imsave(args.output, output_image)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-ps',
        '--patch_size',
        help='the size of the patches',
        type=int,
        default=9
    )
    parser.add_argument(
        '-o',
        '--output',
        help='the file path to save the output image',
        default='result/output9 random third.png'
    )
    parser.add_argument(
        '--plot-progress',
        help='plot each generated image',
        action='store_true',
        default=False
    )
    parser.add_argument(
        'model_image',
        help='the model image'
    )
    parser.add_argument(
        'model_depth',
        help='the model depth of model image'
    )
    parser.add_argument(
        'image',
        help='the image you want to make depth image with'
    )
    return parser.parse_args()


if __name__ == '__main__':
    main()


class Inpainter():
    def __init__(self, model_image, model_depth, image, patch_size, plot_progress=False):
        self.model_image = model_image.astype('uint8')
        self.model_depth = self._to_rgb((model_depth*255).astype('uint8'))
        self.image = image.astype('uint8')

        self.patch_size = patch_size
        self.plot_progress = plot_progress

        # Non initialized attributes
        self.working_image = None
        self.front = None
        self.data = None
        self.priority = None

    def inpaint(self):
        """ Compute the new image and return it """

        self._validate_inputs() # image no size ga same ka check
        self._initialize_attributes()

        start_time = time.time()
        keep_going = True
        while keep_going:
            self._find_front()
            if self.plot_progress:#←推定の様子を示すやつ
                self._plot_image()
            
            priority_start_time = time.time()
            self._update_priority()
            print('_update_priority: %f seconds'
                  % (time.time() - priority_start_time))

            target_pixel = self._find_highest_priority_pixel()
            find_start_time = time.time()
            source_patch = self._find_source_patch(target_pixel)
            print('Time to find best: %f seconds'
                  % (time.time() - find_start_time))

            self._update_image(target_pixel, source_patch)

            keep_going = not self._finished()

        print('Took %f seconds to complete' % (time.time() - start_time))
        return self.output_image

    def _validate_inputs(self):
        if self.model_image.shape[:2] != self.model_depth.shape[:2]:
            raise AttributeError('mask and image must be of the same size')


    def _initialize_attributes(self):
        """ Initialize the non initialized attributes

        The confidence is initially the inverse of the mask, that is, the
        target region is 0 and source region is 1.

        The data starts with zero for all pixels.

        The working image and working mask start as copies of the original
        image and mask.
        """
        model_height, model_width = self.model_image.shape[:2] 
        """"size """
        
        height, width = self.image.shape[:2]

        self.data = np.zeros([height, width])

        self.output_image = self._to_rgb(np.zeros([height, width])).astype('uint8')

        self.working_image = np.copy(self.image)
        self.working_mask = np.ones((height, width)).astype('uint8')
        self.working_mask[:, 0] = 0
        self.working_mask[0, :] = 0
        self.working_mask[:, width-1] = 0
        self.working_mask[height-1, :] = 0

        self.confidence = (1 - self.working_mask).astype(float)

    def _find_front(self):
        """ Find the front using laplacian on the mask

        The laplacian will give us the edges of the mask, it will be positive
        at the higher region (white) and negative at the lower region (black).
        We only want the white region, which is inside the mask, so we
        filter the negative values.
        """  

        self.front = (laplace(self.working_mask) > 0).astype('uint8')
        # self.front = (laplace_numpy(self.working_mask) > 0).astype('uint8')
        # self.front = (laplace_numba(self) > 0).astype('uint8')
        # TODO: check if scipy's laplace filter is faster than scikit's

    

    def _plot_image(self):
        height, width = self.working_mask.shape

        # Remove the target region from the image
        inverse_mask = 1 - self.working_mask
        rgb_inverse_mask = self._to_rgb(inverse_mask)
        image = self.output_image * rgb_inverse_mask


        # Fill the target borders with red
        image[:, :, 0] += self.front * 255

        # Fill the inside of the target region with white
        white_region = (self.working_mask - self.front) * 255
        rgb_white_region = self._to_rgb(white_region)
        image += rgb_white_region

        plt.clf()
        plt.imshow(image)
        plt.draw()
        plt.pause(0.001)  # TODO: check if this is necessary


    def _update_priority(self):
        self._update_confidence()
        self._update_data()
        C=np.array(self.confidence)
        D=np.array(self.data)
        F=np.array(self.front)
        self.priority = C * D * F
        #self.priority = self.confidence * self.data * self.front


    def _update_confidence(self):
        new_confidence = np.copy(self.confidence)
        front_positions = np.argwhere(self.front == 1) # kyoukai bubun wo kiroku
        for point in front_positions:
            patch = self._get_patch(point)
            new_confidence[point[0], point[1]] = sum(sum(
                self._patch_data(self.confidence, patch)
            ))/self._patch_area(patch)

        self.confidence = new_confidence


    def _get_patch(self, point):
        half_patch_size = (self.patch_size-1)//2
        height, width = self.working_image.shape[:2]
        patch = [
            [
                max(0, point[0] - half_patch_size),
                min(point[0] + half_patch_size, height-1)
            ],
            [
                max(0, point[1] - half_patch_size),
                min(point[1] + half_patch_size, width-1)
            ]
        ]
        return patch


    @staticmethod
    def _patch_area(patch):
        return (1+patch[0][1]-patch[0][0]) * (1+patch[1][1]-patch[1][0])


    @staticmethod
    def _patch_data(source, patch):
        return source[
            patch[0][0]:patch[0][1] + 1,
            patch[1][0]:patch[1][1] + 1
        ]


    def _update_data(self):
        normal = self._calc_normal_matrix() # seiki gyouretu
        gradient = self._calc_gradient_matrix() # koubai gyouretu
        no=np.array(normal)
        gr=np.array(gradient)
        
        normal_gradient = no*gr
        #normal_gradient = normal*gradient
        self.data = np.sqrt(
            normal_gradient[:, :, 0]**2 + normal_gradient[:, :, 1]**2
        ) + 0.001  # To be sure to have a greater than 0 data


    def _calc_normal_matrix(self): # seikigyouretu no keisan
        x_kernel = np.array([[.25, 0, -.25], [.5, 0, -.5], [.25, 0, -.25]])
        y_kernel = np.array([[-.25, -.5, -.25], [0, 0, 0], [.25, .5, .25]])

        x_normal = convolve(self.working_mask.astype(float), x_kernel)
        y_normal = convolve(self.working_mask.astype(float), y_kernel)
        normal = np.dstack((x_normal, y_normal))

        height, width = normal.shape[:2]
        yn=np.array(y_normal)
        xn=np.array(x_normal)
        norm = np.sqrt(yn**2 + xn**2) \
                 .reshape(height, width, 1) \
                 .repeat(2, axis=2)#同じこと繰り返し 
        
        #norm = np.sqrt(y_normal**2 + x_normal**2) \
         #        .reshape(height, width, 1) \
          #       .repeat(2, axis=2)#同じこと繰り返し
        norm[norm == 0] = 1

        unit_normal = normal/norm
        return unit_normal


    def _calc_gradient_matrix(self):
        # TODO: find a better method to calc the gradient
        height, width = self.working_image.shape[:2]

        grey_image = rgb2gray(self.working_image)
        grey_image[self.working_mask == 1] = None

        gradient = np.nan_to_num(np.array(np.gradient(grey_image)))
        gradient_val = np.sqrt(gradient[0]**2 + gradient[1]**2)
        max_gradient = np.zeros([height, width, 2])

        front_positions = np.argwhere(self.front == 1)
        for point in front_positions:
            patch = self._get_patch(point)
            patch_y_gradient = self._patch_data(gradient[0], patch)
            patch_x_gradient = self._patch_data(gradient[1], patch)
            patch_gradient_val = self._patch_data(gradient_val, patch)

            patch_max_pos = np.unravel_index(
                patch_gradient_val.argmax(),
                patch_gradient_val.shape
            )

            max_gradient[point[0], point[1], 0] = \
                patch_y_gradient[patch_max_pos]
            max_gradient[point[0], point[1], 1] = \
                patch_x_gradient[patch_max_pos]

        return max_gradient


    def _find_highest_priority_pixel(self):
        point = np.unravel_index(self.priority.argmax(), self.priority.shape)
        return point


    def _find_source_patch(self, target_pixel):
        target_patch = self._get_patch(target_pixel) # 優先度が高いピクセルをtarget_pixelとし，パッチ取得
        height, width = self.working_image.shape[:2]
        model_height, model_width = self.model_image.shape[:2]

        patch_height, patch_width = self._patch_shape(target_patch)

        best_match = None
        best_match_difference = 0

        lab_image = rgb2lab(self.working_image) #lab変換
        lab_model_image = rgb2lab(self.model_image)

       # source_patchをランダムに決める 
        for pnum in range(int(height * width / 100)):
            y = np.random.randint(0, self.model_image.shape[0]-patch_height)
            x = np.random.randint(0, self.model_image.shape[1]-patch_width)
            source_patch = [
                    [y, y + patch_height-1],
                    [x, x + patch_width-1]
                ]
            if source_patch[0][1] >= model_height or source_patch[1][1] >= model_width:
                break
            difference = self._calc_patch_difference(
                    lab_image,
                    lab_model_image,
                    target_patch,
                    source_patch
                )
            if best_match is None or difference < best_match_difference:
                    best_match = source_patch
                    best_match_difference = difference
            if difference == 0:
                    return best_match

        '''for y in range(height - patch_height + 1):#作業画像の範囲←ここをランダムにする
            for x in range(width - patch_width + 1):
                source_patch = [
                    [y, y + patch_height-1],
                    [x, x + patch_width-1]
                ]

                if source_patch[0][1] >= model_height or source_patch[1][1] >= model_width:
                    break

                difference = self._calc_patch_difference(
                    lab_image,
                    lab_model_image,
                    target_patch,
                    source_patch
                )

                if best_match is None or difference < best_match_difference:
                    best_match = source_patch
                    best_match_difference = difference

                if difference == 0:
                    return best_match'''
        return best_match


    def _calc_patch_difference(self, image, model_image, target_patch, source_patch):
        target_data = self._patch_data(
            image,
            target_patch
        )
        source_data = self._patch_data(
            model_image,
            source_patch
        )
        absolute_distance=(np.abs(target_data - source_data)).sum() #誤差の絶対値
        squared_distance = ((target_data - source_data)**2).sum() #平均二乗誤差
        euclidean_distance = np.sqrt(
            (target_patch[0][0] - source_patch[0][0])**2 +
            (target_patch[1][0] - source_patch[1][0])**2
        )  # tie-breaker factor
        #return squared_distance + euclidean_distance
        return squared_distance
        #return absolute_distance


    def _update_image(self, target_pixel, source_patch):
        target_patch = self._get_patch(target_pixel)# 優先度が高いピクセルをtarget_pixelとし，パッチ取得
        pixels_positions = np.argwhere(
            self._patch_data(
                self.working_mask,
                target_patch
            ) == 1
        ) + [target_patch[0][0], target_patch[1][0]]
        patch_confidence = self.confidence[target_pixel[0], target_pixel[1]] #これなに
        for point in pixels_positions:
            self.confidence[point[0], point[1]] = patch_confidence #マスク上の範囲でのCを更新

        mask = self._patch_data(self.working_mask, target_patch)
        rgb_mask = self._to_rgb(mask)
        source_data = self._patch_data(self.model_depth, source_patch)
        target_data = self._patch_data(self.working_image, target_patch)

        #new_data = source_data * rgb_mask + target_data * (1 - rgb_mask)
        new_data = source_data

        self._copy_to_patch(
            self.output_image,
            target_patch,
            new_data
        )
        self._copy_to_patch(
            self.working_mask,
            target_patch,
            0
        )


    def _finished(self):
        height, width = self.working_image.shape[:2]
        remaining = self.working_mask.sum()
        total = height * width
        print('%d of %d completed' % (total-remaining, total))
        return remaining == 0


    @staticmethod
    def _patch_shape(patch):
        return (1+patch[0][1]-patch[0][0]), (1+patch[1][1]-patch[1][0])


    @staticmethod
    def _patch_data(source, patch):
        return source[
            patch[0][0]:patch[0][1]+1,
            patch[1][0]:patch[1][1]+1
        ]


    @staticmethod
    def _to_rgb(image):
        height, width = image.shape
        return image.reshape(height, width, 1).repeat(3, axis=2)


    @staticmethod
    def _copy_to_patch(dest, dest_patch, data):
        dest[
            dest_patch[0][0]:dest_patch[0][1]+1,
            dest_patch[1][0]:dest_patch[1][1]+1
        ] = data

class Median():
    def __init__(self, image):
        self.image = image

    def median(self):
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        median = cv2.medianBlur(gray, ksize=3)
        dst = cv2.cvtColor(median, cv2.COLOR_GRAY2RGB)
        return dst