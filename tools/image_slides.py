import cv2

def image_slide(image_path, output_path):
    image = cv2.imread(image_path)

    line_positions = [4, 13, 22, 31, 40, 49]
    for x in line_positions:
        cv2.line(image, (x, 0), (x, image.shape[0]), (0, 255, 0), 1)

    cv2.imwrite(output_path, image)
    return image

image_concat_list = [[], [], [], [], [], ]
for i in range(25):
    image_path = '../sampleCaptchas/input/input{:02}.jpg'.format(i)
    output_path = 'input_slides/input{:02}.jpg'.format(i)
    # import pdb; pdb.set_trace()
    image_candidate = image_slide(image_path, output_path)
    image_concat_list[i%5].append(image_candidate)

image_concat_list_tmp = []
for j in range(5):
    img_tmp = cv2.hconcat(image_concat_list[j])
    image_concat_list_tmp.append(img_tmp)
image_concat = cv2.vconcat(image_concat_list_tmp)
cv2.imwrite('input_image_slide_concat.jpg', image_concat)

