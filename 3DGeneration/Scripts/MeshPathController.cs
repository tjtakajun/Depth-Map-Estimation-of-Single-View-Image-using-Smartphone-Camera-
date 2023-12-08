using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class MeshPathController : MonoBehaviour
{
    public GameObject DepthCursor;
    public GameObject MeshObject; // MeshObject�ɐ[�x�摜���琶���������b�V����ݒ肷��
    public Material MeshMaterial;
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

        // MeshObject�𐶐�
        GameObject meshObject = Instantiate(MeshObject, pos, Quaternion.identity);

        // DepthCursor�̈ʒu���班����O�Ɉړ�������
        Vector3 toCamera = DepthSource.ARCamera.transform.position - DepthCursor.transform.position;
        meshObject.transform.position = DepthCursor.transform.position + (toCamera.normalized * _avatarOffset);

        // MeshObject��L���ɂ���
        meshObject.SetActive(true);

        _firstWaypointPlaced = true;
    }

    public void Clear()
    {
        if (_root != null)
        {
            foreach (Transform child in _root.transform)
            {
                Destroy(child.gameObject);
            }
        }
    }

    private void OnDestroy()
    {
        Destroy(_root);
        _root = null;
    }

    private void Start()
    {
        // DepthMapToMesh �X�N���v�g�����I�u�W�F�N�g���擾
        DepthMapToMesh depthMapToMesh = FindObjectOfType<DepthMapToMesh>();
        if (depthMapToMesh == null)
        {
            Debug.LogError("DepthMapToMesh script not found in the scene.");
            return;
        }

        // DepthMapToMesh�Ő����������b�V����MeshObject�ɐݒ�
        MeshObject = depthMapToMesh.gameObject;

        // MeshObject �������������b�V���I�u�W�F�N�g�Ƀ}�e���A����ݒ�
        MeshRenderer meshRenderer = MeshObject.GetComponent<MeshRenderer>();
        meshRenderer.material = MeshMaterial;
    }

    private void Update()
    {
        if (!_firstWaypointPlaced)
        {
            Vector3 toCamera = DepthSource.ARCamera.transform.position - DepthCursor.transform.position;
            MeshObject.transform.position = DepthCursor.transform.position + (toCamera.normalized * _avatarOffset);
        }
    }
}
