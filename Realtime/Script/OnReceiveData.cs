using System.Collections;
using System.Collections.Generic;
using UnityEngine;

    public class OnReceiveData : MonoBehaviour
    {
        // Capture�V�[������󂯎�����f�[�^���g�p����
        public void OnReceive(Texture2D yourImage)
        {
            // Image Cropper Prefab���擾���܂�
            //ImageCropper imageCropper = GetComponent<ImageCropper>();

            // Texture2D����Sprite���쐬���AImage�R���|�[�l���g�ɐݒ肷��
            //var sprite = Sprite.Create(yourImage, new Rect(0, 0, yourImage.width, yourImage.height), Vector2.zero);
            //Texture texture = sprite.texture;

            //ImageCropper.Instance.OrientedImage = yourImage;

            // �摜��\�����܂�
            ImageCropperRE.CropResult onCrop = (bool result, Texture originalImage, Texture2D croppedImage) =>
            {
                // �؂����Ƃ����������ꍇ
                if (result)
                {
                    ImageCropperRE.Instance.Crop();
                    // �؂������摜���g�p���܂�
                    // �����ł͗�Ƃ��ĉ摜��\������Image�R���|�[�l���g�ɐݒ肵�܂�

                }
                else
                {
                    //ImageCropper.Instance.Cancel();
                    // �؂����Ƃ��L�����Z�����ꂽ�ꍇ�̏������L�q���܂�
                }
            };

            // �摜��\�����܂�
            ImageCropperRE.Instance.Show(yourImage, onCrop, null, null);

        }
    }

