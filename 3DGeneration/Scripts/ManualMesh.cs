using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using UnityEngine.SceneManagement;
using UnityEngine.XR.ARFoundation;
using UnityEngine.XR.ARSubsystems;

public class ManualMesh : MonoBehaviour
{
    public Texture2D depthImage; // �蓮�ŏ������������摜��Inspector����A�^�b�`����

    private Mesh _mesh;
    public RawImage depth;
    private GameObject ThreeDRecon;
    private List<Vector3> _vertices = new List<Vector3>();
    private List<Color32> _colors = new List<Color32>();
    private List<int> _indices = new List<int>();

    // private void Start()
    //{
    //     DepthMap = GameObject.Find("DepthMap");
    // }

    public GameObject InitializePointCloud()
    {
        // ���b�V�����쐬
        _mesh = new Mesh();
        _mesh.indexFormat = UnityEngine.Rendering.IndexFormat.UInt32;

        // �摜���e�N�X�`���Ƃ��ă}�e���A���ɐݒ�

        // ���b�V���̐���


        // �_�Q�̃��b�V�����쐬����AR��Ԃɔz�u
        GameObject pointCloudObject = new GameObject("PointCloud");
        MeshFilter meshFilter = pointCloudObject.AddComponent<MeshFilter>();
        MeshRenderer meshRenderer = pointCloudObject.AddComponent<MeshRenderer>();
        meshFilter.mesh = _mesh;


        // �|�C���g�N���E�h�I�u�W�F�N�g�̈ʒu��AR��Ԃɍ��킹��
        pointCloudObject.transform.position = transform.position;
        pointCloudObject.transform.rotation = transform.rotation;

        return pointCloudObject; // ���������_�Q�I�u�W�F�N�g��Ԃ�
    }

    public void PickDepthMap()
    {
        // �摜�̓ǂݍ���
        NativeGallery.Permission permission = NativeGallery.GetImageFromGallery((path) =>
        {
            UnityEngine.Debug.Log("Image path: " + path);
            if (path != null)
            {
                // �摜�p�X����Texture2D�𐶐�
                Texture2D texture = NativeGallery.LoadImageAtPath(path, 512);
                if (texture == null)
                {
                    UnityEngine.Debug.Log("Couldn't load texture from " + path);
                    return;
                }

                // RawImage�̌��e�N�X�`����j��
                Destroy(depth.texture);

                // RawImage�ŐV�K�e�N�X�`����\��
                depth.texture = texture;
                depthImage = createReadabeTexture2D(texture);
            }
        });
        UnityEngine.Debug.Log("Permission result: " + permission);
    }

    Texture2D createReadabeTexture2D(Texture2D texture2d)
    {
        RenderTexture renderTexture = RenderTexture.GetTemporary(
                    texture2d.width,
                    texture2d.height,
                    0,
                    RenderTextureFormat.Default,
                    RenderTextureReadWrite.Linear);

        Graphics.Blit(texture2d, renderTexture);
        RenderTexture previous = RenderTexture.active;
        RenderTexture.active = renderTexture;
        Texture2D readableTextur2D = new Texture2D(texture2d.width, texture2d.height);
        readableTextur2D.ReadPixels(new Rect(0, 0, renderTexture.width, renderTexture.height), 0, 0);
        readableTextur2D.Apply();
        RenderTexture.active = previous;
        RenderTexture.ReleaseTemporary(renderTexture);
        return readableTextur2D;
    }



    void MeshGen()
    {
        

    }

    // �P�x�l���Ԃ��Ď擾���郁�\�b�h
    private float InterpolateBrightness(Color32[] pixels, int x, int y, int width, int height, float minBrightness, float maxBrightness)
    {
        // ��f���W���摜�͈͊O�̏ꍇ�A��Ԃ��Ȃ��ŋP�x�l�̍ŏ��l��Ԃ�
        if (x < 0 || x >= width || y < 0 || y >= height)
        {
            return minBrightness;
        }

        int index = y * width + x;
        Color32 pixel = pixels[index];

        // RGB�l�̕��ς��P�x�l�Ƃ��Ď擾
        float brightness = (pixel.r + pixel.g + pixel.b) / 3f;

        // �P�x�l��0����1�ɐ��K��
        float normalizedBrightness = Mathf.InverseLerp(minBrightness, maxBrightness, brightness);

        // ���݂̔{����ݒ�i1���傫���l��ݒ肷��ƌ��݂��傫���Ȃ�܂��j
        float thicknessMultiplier = 3f;

        // ���K�����ꂽ�P�x�l�ɔ{����������
        normalizedBrightness *= thicknessMultiplier;

        return normalizedBrightness;
    }
}
