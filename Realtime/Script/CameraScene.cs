using System.Collections;
using System.Collections.Generic;
using UnityEngine.SceneManagement;
using UnityEngine;

public class CaptureScene : MonoBehaviour
{
    // �J��������̉摜���擾���邽�߂�WebCamTexture
    private WebCamTexture webCamTexture;

    public void CameraCapture()
    {
        // WebCamTexture���쐬���ăJ��������̉摜���擾���J�n����
        webCamTexture = new WebCamTexture();
        webCamTexture.Play();

        // �摜��Texture2D�ɕϊ�����
        var texture2D = new Texture2D(webCamTexture.width, webCamTexture.height);
        texture2D.SetPixels(webCamTexture.GetPixels());

        // Realtime�V�[���Ƀf�[�^��n��
        SceneManager.SetActiveScene(SceneManager.GetSceneByName("Realtime"));
        var scene = SceneManager.GetSceneByName("Realtime");
        var rootGameObjects = scene.GetRootGameObjects();
        foreach (var gameObject in rootGameObjects)
        {
            gameObject.SendMessage("OnReceiveData", texture2D);
        }
    }

    // Capture�V�[������󂯎�����f�[�^���g�p����
    public void OnReceiveData(Texture2D yourImage)
    {
        // Image Cropper Prefab���擾���܂�
        ImageCropper imageCropper = GetComponent<ImageCropper>();

        // �摜��\�����܂�
        ImageCropper.CropResult onCrop = (bool result, Texture originalImage, Texture2D croppedImage) =>
        {
            // �؂����Ƃ����������ꍇ
            if (result)
            {
                // �؂������摜���g�p���܂�
                // �����ł͗�Ƃ��ĉ摜��\������Image�R���|�[�l���g�ɐݒ肵�܂�

            }
            else
            {
                // �؂����Ƃ��L�����Z�����ꂽ�ꍇ�̏������L�q���܂�
            }
        };

        // �摜��\�����܂�
        ImageCropper.Instance.Show(yourImage, onCrop, null, null);

    }
}
