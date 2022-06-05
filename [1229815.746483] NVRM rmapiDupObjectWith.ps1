[1229815.746483] NVRM rmapiDupObjectWithSecInfo: Nv04DupObject: hClient:0xc1d00001 hParent:0xcaf45b0e hObject:0x0
[1229815.746488] NVRM rmapiDupObjectWithSecInfo: Nv04DupObject:  hClientSrc:0xc1d01e25 hObjectSrc:0xcaf00007 flags:0x1
[1229815.746524] NVRM rmapiDupObjectWithSecInfo: ...handle dup complete
[1229815.746603] NVRM rmapiAllocWithSecInfo: client:0xc1d01e25 parent:0xcaf00004 object:0x0 class:0xc56f
[1229815.746622] NVRM kchannelConstruct_IMPL: Not using ctx buf pool
[1229815.746629] NVRM nvAssertFailedNoLog: Assertion failed: numMax == numFree && numMax != 0 @ kernel_channel_group_api.c:838
[1229815.746636] NVRM nvAssertOkFailedNoLog: Assertion failed: Generic Error: Invalid state [NV_ERR_INVALID_STATE] (0x00000040) returned from kchangrpapiSetLegacyMode(pKernelChannelGroupApi, pGpu, pKernelFifo, hClient) @ kernel_channel.c:615
[1229815.746645] NVRM serverAllocResource: hParent 0xcaf00004 : hClass 0x0000c56f allocation failed
[1229815.746648] NVRM rmapiAllocWithSecInfo: allocation failed; status: Generic Error: Invalid state [NV_ERR_INVALID_STATE] (0x00000040)
[1229815.746651] NVRM rmapiAllocWithSecInfo: client:0xc1d01e25 parent:0xcaf00004 object:0xcaf00008 class:0xc56f

[1229855.272314] NVRM rmapiAllocWithSecInfo: client:0xc1d01e2a parent:0x5c000011 object:0x5c000018 class:0xc56f
[1229855.272352] NVRM kchannelConstruct_IMPL: Not using ctx buf pool
[1229855.272454] NVRM _kbusWalkCBMapNextEntries_RmAperture: [GPU0]: PA 0x2F6DD4000, Entries 0xC0-0xC0
[1229855.272468] NVRM kfifoChannelGetFifoContextMemDesc_GM107: Channel 13 engine 0xb0236500 engineState 0x1 *ppMemDesc FFFF9066E66C6820
[1229855.272473] NVRM kchannelAllocMem_GM107: hChannel 0x5c000018 hClient 0xc1d01e2a, Class ID 0xc56f Instance Block @ 0x2f6a3f000 (VIDEO MEMORY 2) USERD @ 0xfc02000 for subdevice 0
[1229855.272476] NVRM kchangrpAddChannel_IMPL: Channel 0xd within TSG 0xd is using subcontext 0x3f
[1229855.273276] NVRM _kchannelSendChannelAllocRpc: Alloc Channel chid 13, hClient:0xc1d01e2a, hParent:0x5c000011, hObject:0x5c000018, hClass:0xc56f
[1229855.273293] NVRM rmapiAllocWithSecInfo: allocation complete


[1229924.246064] NVRM pmaAllocatePages: Attempt discontiguous allocation of 0x1 pages of size 0x200000 (0x20 frames per page)
[1229924.246065] NVRM pmaRegmapScanDiscontiguous: Scanning with addrBase 0x52c0000 in frame range 0x14..0x2e62f, pages to allocate 0x1                                                      
[1229924.246067] NVRM pmaAllocatePages: Successfully allocated frames:                                                                                                                                            
[1229924.246070] NVRM pmaAllocatePages: 0x3d4 through 0x3f3                                     
[1229924.246091] NVRM rmapiControlWithSecInfo: Nv04Control: hClient:0xc1d00001 hObject:0xcaf45b94 cmd:0x801813 params:FFFF9FE64635F920 paramSize:0x20 flags:0x0
[1229924.246092] NVRM _rmapiRmControl: rmControl: hClient 0xc1d00001 hObject 0xcaf45b94 cmd 0x801813
[1229924.246093] NVRM _rmapiRmControl: rmControl: pUserParams 0xFFFF9FE64635F920 paramSize 0x20
[1229924.250130] NVRM _gmmuWalkCBCopyEntries: [GPU0]: GVAS(FFFF906B7E57D020) PA 0x2F6A46000 -> PA 0x9000000, Entries 0x0-0x0
[1229924.250137] NVRM _kbusWalkCBMapNextEntries_RmAperture: [GPU0]: PA 0x2F6DD4000, Entries 0xBF-0xBF
[1229924.250153] NVRM _gmmuWalkCBUpdatePdb: [GPU0]: PA 0x9000000 (valid)                                                                                                                                          
[1229924.250154] NVRM _gmmuWalkCBLevelFree: [GPU0]: PA 0x2F6A46000 for VA 0x0-0x1FFFFFFFFFFFF
[1229924.250156] NVRM rmapiControlWithSecInfo: Nv04Control: control complete
