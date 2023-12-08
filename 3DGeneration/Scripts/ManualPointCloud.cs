using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using UnityEngine.SceneManagement;
using UnityEngine.XR.ARFoundation;
using UnityEngine.XR.ARSubsystems;

public class ManualPointCloud : MonoBehaviour
{
    public Texture2D depthImage; // �蓮�ŏ������������摜��Inspector����A�^�b�`����
    public Material pointCloudMaterial; // �|�C���g�N���E�h�̃}�e���A����Inspector����A�^�b�`����

    private Mesh _mesh;
    public RawImage depth;
    private GameObject DepthMap;
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
        pointCloudMaterial.SetTexture("_DepthTex", depthImage);

        // �|�C���g�N���E�h�̐���
        CalculatePointCloud();

        // �_�Q�̃��b�V�����쐬����AR��Ԃɔz�u
        GameObject pointCloudObject = new GameObject("PointCloud");
        MeshFilter meshFilter = pointCloudObject.AddComponent<MeshFilter>();
        MeshRenderer meshRenderer = pointCloudObject.AddComponent<MeshRenderer>();
        meshFilter.mesh = _mesh;
        meshRenderer.material = pointCloudMaterial;

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



    void CalculatePointCloud()
    {
        Color32[] pixels = depthImage.GetPixels32();

        // �摜�̕��ƍ������擾
        int width = depthImage.width;
        int height = depthImage.height;

        // �V�����_�Q�̕��ƍ�����ݒ�
        int newWidth = width * 2;
        int newHeight = height * 2;

        // �P�x�l�̍ŏ��l�ƍő�l���擾
        float minBrightness = float.MaxValue;
        float maxBrightness = float.MinValue;

        // �P�x�l�̍ŏ��l�ƍő�l������
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                int index = y * width + x;
                Color32 pixel = pixels[index];

                // RGB�l�̕��ς��P�x�l�Ƃ��Ď擾
                float brightness = (pixel.r + pixel.g + pixel.b) / 3f;

                minBrightness = Mathf.Min(minBrightness, brightness);
                maxBrightness = Mathf.Max(maxBrightness, brightness);
            }
        }

        // �|�C���g�N���E�h�̐���
        for (int y = 0; y < newHeight; y++)
        {
            for (int x = 0; x < newWidth; x++)
            {
                // ��ԑΏۂ̉�f���W���v�Z
                int srcX = Mathf.FloorToInt(x / 2f);
                int srcY = Mathf.FloorToInt(y / 2f);

                // �P�x�l���Ԃ��Ď擾
                float normalizedBrightness = InterpolateBrightness(pixels, srcX, srcY, width, height, minBrightness, maxBrightness);

                // 3D��Ԃ̍��W���v�Z�i�ȒP��x����y���̓s�N�Z�����W�Az���͐��K�����ꂽ�P�x�l�Ƃ���j
                Vector3 vertex = new Vector3(x, y, normalizedBrightness);

                // �|�C���g�N���E�h�̒��_���X�g�ƐF��񃊃X�g�ɒǉ�
                _vertices.Add(vertex);
                _colors.Add(pixels[srcY * width + srcX]);

                // �C���f�b�N�X����ǉ�
                _indices.Add(_vertices.Count - 1);
            }
        }

        Debug.Log("Vertex count: " + _vertices.Count);
        for (int i = 0; i < _vertices.Count; i++)
        {
            Debug.Log("Vertex " + i + ": " + _vertices[i] + ", Color: " + _colors[i]);
        }

        // ���b�V���̒��_���ƐF����ݒ�
        _mesh.SetVertices(_vertices);
        _mesh.SetColors(_colors);
        _mesh.SetIndices(_indices.ToArray(), MeshTopology.Points, 0);
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
        float thicknessMultiplier = 21f;

        // ���K�����ꂽ�P�x�l�ɔ{����������
        normalizedBrightness *= thicknessMultiplier;

        return normalizedBrightness;
    }
}
