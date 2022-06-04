typedef void* RsResourceRef;
typedef void* API_SECURITY_INFO;

struct RS_LOCK_INFO
{
    struct RsClient *pClient;              ///< Pointer to client that was locked (if any)
    struct RsClient *pSecondClient;        ///< Pointer to second client, for dual-client locking
    RsResourceRef *pContextRef;     ///< User-defined reference
    struct RsSession *pSession;            ///< Session object to be locked, if any
    NvU32 flags;                    ///< RS_LOCK_FLAGS_*
    NvU32 state;                    ///< RS_LOCK_STATE_*
    NvU32 gpuMask;
    NvU8  traceOp;                  ///< RS_LOCK_TRACE_* operation for lock-metering
    NvU32 traceClassId;             ///< Class of initial resource that was locked for lock metering
};

struct RS_RES_ALLOC_PARAMS_INTERNAL
{
    NvHandle hClient;       ///< [in] The handle of the resource's client
    NvHandle hParent;       ///< [in] The handle of the resource's parent. This may be a client or another resource.
    NvHandle hResource;     ///< [inout] Server will assign a handle if this is 0, or else try the value provided
    NvU32 externalClassId;  ///< [in] External class ID of resource
    NvHandle hDomain;       ///< UNUSED

    // Internal use only
    RS_LOCK_INFO           *pLockInfo;        ///< [inout] Locking flags and state
    struct RsClient               *pClient;          ///< [out] Cached client
    RsResourceRef          *pResourceRef;     ///< [out] Cached resource reference
    NvU32                   allocFlags;       ///< [in] Allocation flags
    NvU32                   allocState;       ///< [inout] Allocation state
    API_SECURITY_INFO      *pSecInfo;

    void                   *pAllocParams;     ///< [in] Copied-in allocation parameters

    // ... Dupe alloc
    struct RsClient               *pSrcClient;       ///< The client that is sharing the resource
    RsResourceRef          *pSrcRef;          ///< Reference to the resource that will be shared

    RS_ACCESS_MASK         *pRightsRequested; ///< [in] Access rights requested on the new resource
    // Buffer for storing contents of user mask. Do not use directly, use pRightsRequested instead.
    RS_ACCESS_MASK          rightsRequestedCopy;

    RS_ACCESS_MASK         *pRightsRequired;  ///< [in] Access rights required to alloc this object type
};