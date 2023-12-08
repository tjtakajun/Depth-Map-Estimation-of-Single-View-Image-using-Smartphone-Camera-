using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ThreeDMeshGen : MonoBehaviour
{
    public GameObject DepthCursor;
    public GameObject PointCloud;
    public Material PointCloudMaterial;
    public GameObject Waypoint;

    private const float _avatarOffset = 0.9f;
    private const float _waypointYOffset = 0.05f;
    private GameObject _root;
    private bool _firstWaypointPlaced;

    public void DropWaypoint()
    {
        if (_root == null)
        {
            _root = new GameObject("Waypoints");
        }

        Vector3 pos = DepthCursor.transform.position;
        pos.y += _waypointYOffset;

        // PointCloud�I�u�W�F�N�g�𐶐�
        GameObject pointCloudObject = Instantiate(PointCloud, pos, Quaternion.identity);

        // DepthCursor�̈ʒu���班����O�Ɉړ�������
        Vector3 toCamera = DepthSource.ARCamera.transform.position - DepthCursor.transform.position;
        pointCloudObject.transform.position = DepthCursor.transform.position + (toCamera.normalized * _avatarOffset);

        // PointCloud�I�u�W�F�N�g��L���ɂ���
        pointCloudObject.SetActive(true);

        _firstWaypointPlaced = true;

        // �|�C���g�N���E�h���������ꂽ���ǂ����̃��O���o��
        if (pointCloudObject != null)
        {
            Debug.Log("PointCloud object created successfully at DepthCursor position.");
        }
        else
        {
            Debug.LogError("Failed to create PointCloud object at DepthCursor position.");
        }
    }


    public void Clear()
    {
        if (_root != null)
        {
            foreach (Transform child in _root.transform)
            {
                Destroy(child.gameObject);
            }
            _firstWaypointPlaced = false; // �|�C���g�N���E�h���N���A�����̂ŁA_firstWaypointPlaced�����Z�b�g����
        }
    }

    private void OnDestroy()
    {
        Destroy(_root);
        _root = null;
    }

    private void Start()
    {
        // ManualPointCloud �X�N���v�g�����I�u�W�F�N�g���擾
        
        //if (_manualPointCloud == null)
        //{
         //   Debug.LogError("ManualPointCloud script not found in the scene.");
         //   return;
        //}

       // PointCloud = _manualPointCloud.InitializePointCloud(); // ���������\�b�h���Ăяo��

        // ManualPointCloud �����������_�Q�I�u�W�F�N�g�Ƀ}�e���A����ݒ�
        MeshRenderer meshRenderer = PointCloud.GetComponent<MeshRenderer>();
        meshRenderer.material = PointCloudMaterial;
        // �_�Q���������ꂽ���ǂ����̃��O���o��
        if (PointCloud != null)
        {
            Debug.Log("PointCloud object created successfully.");
        }
        else
        {
            Debug.LogError("Failed to create PointCloud object.");
        }
        // �|�C���g�N���E�h�I�u�W�F�N�g�̃X�P�[���𒲐�
        float pointCloudScale = 0.030f; // ��Ƃ��� 0.001 ���X�P�[���Ƃ��Ďg�p
        PointCloud.transform.localScale = new Vector3(pointCloudScale, pointCloudScale, pointCloudScale);

    }


    public void NewPointCloud()
    {
        var start_time = DateTime.Now;
        // ManualPointCloud �X�N���v�g�����I�u�W�F�N�g���擾
        //_manualPointCloud = FindObjectOfType<ManualPointCloud>();
        //PointCloud = _manualPointCloud.InitializePointCloud(); // �摜�ύX���\�b�h���Ăяo��

        // ManualPointCloud �����������_�Q�I�u�W�F�N�g�Ƀ}�e���A����ݒ�
        MeshRenderer meshRenderer = PointCloud.GetComponent<MeshRenderer>();
        meshRenderer.material = PointCloudMaterial;
        // �_�Q���������ꂽ���ǂ����̃��O���o��
        if (PointCloud != null)
        {
            Debug.Log("Mesh object created successfully 2.");
        }
        else
        {
            Debug.LogError("Failed to create Mesh object 2.");
        }
        // ���b�V���I�u�W�F�N�g�̃X�P�[���𒲐�
        float pointCloudScale = 0.030f; // ��Ƃ��� 0.001 ���X�P�[���Ƃ��Ďg�p
        PointCloud.transform.localScale = new Vector3(pointCloudScale, pointCloudScale, pointCloudScale);
        _firstWaypointPlaced = false;
        UnityEngine.Debug.Log(String.Format("Took {0} seconds to complete", DateTime.Now - start_time));
    }

    private void Update()
    {
        if (!_firstWaypointPlaced)
        {
            Vector3 toCamera = DepthSource.ARCamera.transform.position - DepthCursor.transform.position;
            PointCloud.transform.position = DepthCursor.transform.position + (toCamera.normalized * _avatarOffset);
            // 90�x��]���邽�߂̃N�H�[�^�j�I�����쐬
            //Quaternion rotation = Quaternion.Euler(0f, 90f, 0f);

            // PointCloud�I�u�W�F�N�g����]������
            // PointCloud.transform.rotation *= rotation;
        }
    }
}
