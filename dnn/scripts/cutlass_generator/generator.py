#
# \file generator.py
#
# \brief Generates the CUTLASS Library's instances
#

import enum
import os.path
import shutil
import argparse

from library import *
from manifest import *
###################################################################################################

#
def CudaToolkitVersionSatisfies(semantic_ver_string, major, minor, patch = 0):

  # by default, use the latest CUDA Toolkit version
  cuda_version = [11, 0, 132]

  # Update cuda_version based on parsed string
  if semantic_ver_string != '':
    for i, x in enumerate([int(x) for x in semantic_ver_string.split('.')]):
      if i < len(cuda_version):
        cuda_version[i] = x
      else:
        cuda_version.append(x)
  return cuda_version >= [major, minor, patch]


###################################################################################################
###################################################################################################

#
def CreateGemmOperator(manifest, layouts, tile_descriptions, data_type, \
  alignment_constraints, complex_transforms = None, epilogue_functor = EpilogueFunctor.LinearCombination, \
  swizzling_functor = SwizzlingFunctor.Identity8):

  if complex_transforms is None:
    complex_transforms = [(ComplexTransform.none, ComplexTransform.none),]

  element_a, element_b, element_c, element_epilogue = data_type
  
  operations = []

  # by default, only generate the largest tile and largest alignment
  if manifest.args.kernels == '':
    tile_descriptions = [tile_descriptions[0],]
    alignment_constraints = [alignment_constraints[0],]

  for layout in layouts:
    for tile_description in tile_descriptions:
      for alignment in alignment_constraints:
        for complex_transform in complex_transforms:
  
            alignment_c = min(8, alignment)
 
            A = TensorDescription(element_a, layout[0], alignment, complex_transform[0])
            B = TensorDescription(element_b, layout[1], alignment, complex_transform[1])
            C = TensorDescription(element_c, layout[2], alignment_c)

            new_operation = GemmOperation(GemmKind.Universal, tile_description.minimum_compute_capability, \
              tile_description, A, B, C, element_epilogue, epilogue_functor, swizzling_functor)

            manifest.append(new_operation)
            operations.append(new_operation)

  return operations

###########################################################################################################
#   ConvolutionOperator support variations
#        ____________________________________________________________________
#         ConvolutionalOperator |        Analytic      |      Optimized
#        ____________________________________________________________________
#        |       Fprop          |     (strided)        |    (strided)
#        |       Dgrad          |   (strided, unity*)  |     (unity)
#        |       Wgrad          |     (strided)        |    (strided)
#        ____________________________________________________________________
#
# Note :  Operator marked (*) are supported but not generated to keep the instantiated kernel count low
###########################################################################################################
# Convolution for 2D operations
def CreateConv2dOperator(manifest, layout, tile_descriptions, data_type, alignment, \
  conv_kinds = [ConvKind.Fprop, ConvKind.Dgrad, ConvKind.Wgrad], epilogue_functor = EpilogueFunctor.LinearCombination):
  
  element_a, element_b, element_c, element_epilogue = data_type
  
  # one exceptional case
  alignment_c = min(8, alignment)
  
  # iterator algorithm (analytic and optimized)
  iterator_algorithms = [IteratorAlgorithm.Analytic, IteratorAlgorithm.Optimized]

  # by default, only generate the largest tile size
  if manifest.args.kernels == '':
    tile_descriptions = [tile_descriptions[0],]

  operations = []

  for tile in tile_descriptions:
    for conv_kind in conv_kinds:
      for iterator_algorithm in iterator_algorithms:
        A = TensorDescription(element_a, layout[0], alignment)
        B = TensorDescription(element_b, layout[1], alignment)
        C = TensorDescription(element_c, layout[2], alignment_c)

        # unity stride only for Optimized Dgrad
        if (iterator_algorithm == IteratorAlgorithm.Optimized) and (conv_kind == ConvKind.Dgrad):
          new_operation = Conv2dOperation(conv_kind, iterator_algorithm, tile.minimum_compute_capability, tile,\
            A, B, C, element_epilogue, StrideSupport.Unity, epilogue_functor)

          manifest.append(new_operation)
          operations.append(new_operation)

        # strided dgrad is not supported by Optimized Dgrad
        if (iterator_algorithm == IteratorAlgorithm.Optimized) and (conv_kind == ConvKind.Dgrad):
          continue 

        # strided support for Fprop (Analytic/Optimized), Dgrad (Analytic), and Wgrad (Analytic)
        new_operation = Conv2dOperation(conv_kind, iterator_algorithm, tile.minimum_compute_capability, tile,\
         A, B, C, element_epilogue, StrideSupport.Strided, epilogue_functor)

        manifest.append(new_operation)
        operations.append(new_operation)

  return operations

###################################################################################################
###################################################################################################

def GenerateConv2d_Simt(args):
  operations = []

  layouts = [
    (LayoutType.TensorNC4HW4, LayoutType.TensorC4RSK4), 
  ]
    
  math_instructions = [
    MathInstruction(                                  \
      [1, 1, 4],                                      \
      DataType.s8, DataType.s8, DataType.s32,         \
      OpcodeClass.Simt,                               \
      MathOperation.multiply_add),
  ]

  dst_layouts = [
      LayoutType.TensorNC4HW4, 
      LayoutType.TensorNC32HW32, 
      LayoutType.TensorNHWC, 
      LayoutType.TensorNHWC, 
      LayoutType.TensorNCHW
  ]

  dst_types = [
      DataType.s8, 
      DataType.s8, 
      DataType.u4, 
      DataType.s4, 
      DataType.f32, 
  ]

  max_cc = 1024

  for math_inst in math_instructions:
    for layout in layouts:
      for dst_type, dst_layout in zip(dst_types, dst_layouts):
        if dst_type == DataType.s4 or dst_type == DataType.u4:
          min_cc = 75
          skip_unity_kernel = True
        else:
          min_cc = 61
          skip_unity_kernel = False
        tile_descriptions = [
          TileDescription([128, 128, 32], 2, [2, 4, 1], math_inst, min_cc, max_cc),
          TileDescription([128,  64, 32], 2, [2, 2, 1], math_inst, min_cc, max_cc),
          TileDescription([ 64, 128, 32], 2, [1, 4, 1], math_inst, min_cc, max_cc),
          TileDescription([ 64,  64, 32], 2, [1, 2, 1], math_inst, min_cc, max_cc),
          TileDescription([128,  32, 32], 2, [2, 1, 1], math_inst, min_cc, max_cc),
          TileDescription([ 32, 128, 32], 2, [1, 2, 1], math_inst, min_cc, max_cc),
          TileDescription([ 32,  64, 32], 2, [1, 1, 1], math_inst, min_cc, max_cc), 
          TileDescription([ 64,  32, 32], 2, [1, 1, 1], math_inst, min_cc, max_cc), 
          TileDescription([ 32,  32, 32], 2, [1, 1, 1], math_inst, min_cc, max_cc), 
          TileDescription([ 16, 128, 16], 1, [1, 1, 1], math_inst, min_cc, max_cc), 
          TileDescription([ 16,  64,  8], 2, [1, 1, 1], math_inst, min_cc, max_cc), 
        ] 
        operations += GenerateConv2d(ConvKind.Fprop, tile_descriptions, layout[0], layout[1], 
                                     dst_layout, dst_type, min_cc, 32, 32, 32, 
                                     skip_unity_kernel) 
  return operations


def GenerateConv2d_TensorOp_8816(args):
  operations = []

  layouts = [
    (LayoutType.TensorNC32HW32, LayoutType.TensorC32RSK32), 
  ]
    
  math_instructions = [
    MathInstruction(                                  \
      [8, 8, 16],                                     \
      DataType.s8, DataType.s8, DataType.s32,         \
      OpcodeClass.TensorOp,                           \
      MathOperation.multiply_add_saturate),
  ]

  dst_layouts = [
      LayoutType.TensorNC32HW32, 
      LayoutType.TensorNC4HW4, 
  ]

  dst_types = [
      DataType.s8, 
      DataType.s8, 
  ]

  min_cc = 75
  max_cc = 1024

  for math_inst in math_instructions:
    for layout in layouts:
      for dst_type, dst_layout in zip(dst_types, dst_layouts):
        if dst_layout == LayoutType.TensorNC32HW32:
          tile_descriptions = [
            TileDescription([256, 128, 64], 2, [4, 2, 1], math_inst, min_cc, max_cc),
            TileDescription([128, 256, 64], 2, [2, 4, 1], math_inst, min_cc, max_cc),
            TileDescription([128, 128, 64], 2, [2, 2, 1], math_inst, min_cc, max_cc),
            TileDescription([ 64, 128, 64], 2, [2, 2, 1], math_inst, min_cc, max_cc),
            TileDescription([128,  64, 64], 2, [2, 2, 1], math_inst, min_cc, max_cc),
            TileDescription([ 64,  64, 64], 2, [2, 2, 1], math_inst, min_cc, max_cc),
            TileDescription([ 32,  64, 64], 2, [1, 4, 1], math_inst, min_cc, max_cc),
          ] 
        else:
          assert dst_layout == LayoutType.TensorNC4HW4
          tile_descriptions = [
            TileDescription([256, 128, 64], 2, [4, 2, 1], math_inst, min_cc, max_cc),
            TileDescription([128, 256, 64], 2, [2, 4, 1], math_inst, min_cc, max_cc),
            TileDescription([128, 128, 64], 2, [2, 2, 1], math_inst, min_cc, max_cc),
            TileDescription([ 64, 128, 64], 2, [2, 2, 1], math_inst, min_cc, max_cc),
            TileDescription([128,  64, 64], 2, [2, 2, 1], math_inst, min_cc, max_cc),
            TileDescription([ 64,  64, 64], 2, [2, 2, 1], math_inst, min_cc, max_cc),
            TileDescription([ 32,  64, 64], 2, [2, 2, 1], math_inst, min_cc, max_cc),
          ]
        operations += GenerateConv2d(ConvKind.Fprop, tile_descriptions, layout[0], layout[1], 
                                     dst_layout, dst_type, min_cc, 128, 128, 64,  
                                     False) 
  return operations

def GenerateConv2d_TensorOp_8832(args):
  operations = []

  layouts = [
    (LayoutType.TensorNC64HW64, LayoutType.TensorC64RSK64), 
  ]
    
  math_instructions = [
    MathInstruction(                                  \
      [8, 8, 32],                                     \
      DataType.s4, DataType.s4, DataType.s32,         \
      OpcodeClass.TensorOp,                           \
      MathOperation.multiply_add_saturate),           \
    MathInstruction(                                  \
      [8, 8, 32],                                     \
      DataType.s4, DataType.u4, DataType.s32,         \
      OpcodeClass.TensorOp,                           \
      MathOperation.multiply_add_saturate)
  ]

  dst_layouts = [
      LayoutType.TensorNC64HW64, 
  ]

  min_cc = 75
  max_cc = 1024

  for math_inst in math_instructions:
    for layout in layouts:
      for dst_layout in dst_layouts:
        dst_type = math_inst.element_b
        tile_descriptions = [
          TileDescription([256, 128, 128], 2, [4, 2, 1], math_inst, min_cc, max_cc),
          TileDescription([128, 128, 128], 2, [2, 2, 1], math_inst, min_cc, max_cc),
        ] 
        operations += GenerateConv2d(ConvKind.Fprop, tile_descriptions, layout[0], layout[1], 
                                     dst_layout, dst_type, min_cc, 128, 128, 64,  
                                     True) 

  layouts_nhwc = [
    (LayoutType.TensorNHWC, LayoutType.TensorNC8HW8, 32), 
    (LayoutType.TensorNHWC, LayoutType.TensorNC16HW16, 64), 
    (LayoutType.TensorNHWC, LayoutType.TensorNC32HW32, 128), 
  ]

  dst_layouts_nhwc = [
      LayoutType.TensorNHWC, 
  ]

  for math_inst in math_instructions:
    for layout in layouts_nhwc:
      for dst_layout in dst_layouts_nhwc:
        dst_type = math_inst.element_b
        tile_descriptions = [
          TileDescription([128, 32, 64], 2, [2, 1, 1], math_inst, min_cc, max_cc),
          TileDescription([128, 64, 64], 2, [2, 1, 1], math_inst, min_cc, max_cc),
        ] 
        operations += GenerateConv2d(ConvKind.Fprop, tile_descriptions, layout[0], layout[1], 
                                     dst_layout, dst_type, min_cc, layout[2], layout[2], 32,  
                                     False, ImplicitGemmMode.GemmTn) 
  return operations

def GenerateDeconv_Simt(args):
  operations = []

  layouts = [
    (LayoutType.TensorNC4HW4, LayoutType.TensorK4RSC4), 
  ]
    
  math_instructions = [
    MathInstruction(                                  \
      [1, 1, 4],                                      \
      DataType.s8, DataType.s8, DataType.s32,         \
      OpcodeClass.Simt,                               \
      MathOperation.multiply_add),
  ]

  dst_layouts = [
      LayoutType.TensorNC4HW4, 
  ]

  dst_types = [
      DataType.s8, 
  ]

  min_cc = 61
  max_cc = 1024

  for math_inst in math_instructions:
    for layout in layouts:
      for dst_type, dst_layout in zip(dst_types, dst_layouts):
        tile_descriptions = [
          TileDescription([64, 128, 32], 2, [1, 4, 1], math_inst, min_cc, max_cc),
          TileDescription([32, 128, 32], 2, [1, 2, 1], math_inst, min_cc, max_cc),
          TileDescription([16, 128, 16], 2, [1, 2, 1], math_inst, min_cc, max_cc),
          TileDescription([16, 128, 16], 1, [1, 1, 1], math_inst, min_cc, max_cc),
          TileDescription([16,  64,  8], 2, [1, 1, 1], math_inst, min_cc, max_cc),
        ] 
        operations += GenerateConv2d(ConvKind.Dgrad, tile_descriptions, layout[0], layout[1], 
                                     dst_layout, dst_type, min_cc, 32, 32, 32, 
                                     True) 
  return operations

################################################################################
# parameters
# Edge - for tiles, the edges represent the length of one side
# Ratio - the maximum ratio between 2 edges, limits the skinnyness of tiles
# MaxEdge - maximum length of each edge
# Min/Max - minimum/maximum of the product of edge lengths
################################################################################

warpsPerThreadblockEdge = [1, 2, 4, 8, 16]
warpsPerThreadblockRatio = 2
warpsPerThreadblockMax = 16
# NOTE 1x32 and 2x16 warp tile shapes fail validation for ~10% of cases

warpShapeEdges = [8, 16, 32, 64, 128, 256]
warpShapeRatio = 4
warpShapeMax = 64*64
warpShapeMin = 8*8

threadblockEdgeMax = 256

#   char,         type             bits/elem, max tile,    L0 threadblock tiles
precisions = {
    "c" : [ "cutlass::complex<float>",   64,  64*128, [ [ 64, 128], [ 64,  32]             ] ],
    "d" : [ "double",                    64,   64*64, [ [ 64,  64], [ 32,  32]             ] ],
    "h" : [ "cutlass::half_t",           16, 128*256, [ [256, 128], [ 64, 128], [ 64,  32] ] ],
    "i" : [ "int",                       32, 128*128, [ [128,  64], [ 16, 32]              ] ],
    "s" :  [ "float",                     32, 128*128, [ [128, 256], [128, 128], [ 64,  64] ] ],
    "z" : [ "cutlass::complex<double>", 128,   64*64, [ [ 32,  64], [ 16,  32]             ] ],
}
# L1 will have a single kernel for every unique shape
# L2 will have everything else
def GenerateGemm_Simt(args):
  ################################################################################
  # warps per threadblock
  ################################################################################
  warpsPerThreadblocks = []
  for warpsPerThreadblock0 in warpsPerThreadblockEdge:
      for warpsPerThreadblock1 in warpsPerThreadblockEdge:
          if warpsPerThreadblock0 / warpsPerThreadblock1 <= warpsPerThreadblockRatio \
         and warpsPerThreadblock1 / warpsPerThreadblock0 <= warpsPerThreadblockRatio \
         and warpsPerThreadblock0 * warpsPerThreadblock1 <= warpsPerThreadblockMax:
              warpsPerThreadblocks.append([warpsPerThreadblock0,
                  warpsPerThreadblock1])
  
  ################################################################################
  # warp shapes
  ################################################################################
  warpNumThreads = 32
  warpShapes = []
  for warp0 in warpShapeEdges:
      for warp1 in warpShapeEdges:
          if warp0 / warp1 <= warpShapeRatio \
         and warp1 / warp0 <= warpShapeRatio \
         and warp0 * warp1 <= warpShapeMax \
         and warp0*warp1 > warpShapeMin:
              warpShapes.append([warp0, warp1])
  
  # sgemm
  precisionType, precisionBits, threadblockMaxElements, threadblockTilesL0 = precisions["s"]
  
  layouts = [
    (LayoutType.ColumnMajor, LayoutType.ColumnMajor, LayoutType.RowMajor), # nn 
    (LayoutType.ColumnMajor, LayoutType.RowMajor, LayoutType.RowMajor),    # nt
    (LayoutType.RowMajor, LayoutType.ColumnMajor, LayoutType.RowMajor),    # tn
    (LayoutType.RowMajor, LayoutType.RowMajor, LayoutType.RowMajor),       # tt
  ]
  
  math_instructions = [
    MathInstruction(                                  \
      [1, 1, 1],                                      \
      DataType.f32, DataType.f32, DataType.f32,       \
      OpcodeClass.Simt,                               \
      MathOperation.multiply_add),
  ]
  
  min_cc = 50
  max_cc = 1024
 
  operations = []
  for math_inst in math_instructions:
    for layout in layouts:
      data_type = [
        math_inst.element_a,
        math_inst.element_b,
        math_inst.element_accumulator,
        math_inst.element_accumulator,
      ]
      tile_descriptions = [
        TileDescription([64,  256, 8], 2, [2, 4, 1], math_inst, min_cc, max_cc),
        TileDescription([256,  64, 8], 2, [4, 2, 1], math_inst, min_cc, max_cc),
        TileDescription([ 32, 256, 8], 2, [2, 4, 1], math_inst, min_cc, max_cc),
        TileDescription([256,  32, 8], 2, [4, 2, 1], math_inst, min_cc, max_cc),
        TileDescription([128, 128, 8], 2, [4, 2, 1], math_inst, min_cc, max_cc),
        TileDescription([128,  64, 8], 2, [2, 2, 1], math_inst, min_cc, max_cc),
        TileDescription([ 64, 128, 8], 2, [2, 2, 1], math_inst, min_cc, max_cc),
        TileDescription([128,  32, 8], 2, [2, 1, 1], math_inst, min_cc, max_cc),
        TileDescription([ 32, 128, 8], 2, [1, 2, 1], math_inst, min_cc, max_cc),
        TileDescription([ 64,  64, 8], 2, [2, 1, 1], math_inst, min_cc, max_cc),
        TileDescription([ 32,  64, 8], 2, [1, 1, 1], math_inst, min_cc, max_cc),
        TileDescription([ 64,  32, 8], 2, [1, 1, 1], math_inst, min_cc, max_cc),
        TileDescription([ 32,  32, 8], 2, [1, 1, 1], math_inst, min_cc, max_cc),
        TileDescription([  8,  32, 8], 2, [1, 1, 1], math_inst, min_cc, max_cc),
        TileDescription([ 16,  32, 8], 2, [1, 1, 1], math_inst, min_cc, max_cc),
        TileDescription([ 16,  64, 8], 2, [1, 1, 1], math_inst, min_cc, max_cc),
        TileDescription([ 16, 128, 8], 2, [1, 2, 1], math_inst, min_cc, max_cc),
      ]
      for warpsPerThreadblock in warpsPerThreadblocks:
        for warpShape in warpShapes:
          warpThreadsM = 0
          if warpShape[0] > warpShape[1]:
              warpThreadsM = 8
          else:
              warpThreadsM = 4
          warpThreadsN = warpNumThreads / warpThreadsM
  
          # skip shapes with conflicting rectangularity
          # they are unlikely to be fastest
          blockG = warpsPerThreadblock[0] > warpsPerThreadblock[1]
          blockL = warpsPerThreadblock[0] < warpsPerThreadblock[1]
          warpG = warpShape[0] > warpShape[1]
          warpL = warpShape[0] < warpShape[1]
  
          blockG2 = warpsPerThreadblock[0] > warpsPerThreadblock[1]*2
          blockL2 = warpsPerThreadblock[0]*2 < warpsPerThreadblock[1]
          warpG2 = warpShape[0] > warpShape[1]*2
          warpL2 = warpShape[0]*2 < warpShape[1]
  
          if blockG2 and warpL: continue
          if blockL2 and warpG: continue
          if warpG2 and blockL: continue
          if warpL2 and blockG: continue
  
          # check threadblock ratios and max
          threadblockTile = [warpShape[0]*warpsPerThreadblock[0],
                  warpShape[1]*warpsPerThreadblock[1]]
          if threadblockTile[0] * threadblockTile[1] > threadblockMaxElements: continue
          if threadblockTile[0] > threadblockEdgeMax: continue
          if threadblockTile[1] > threadblockEdgeMax: continue
          totalThreads = warpNumThreads*warpsPerThreadblock[0]*warpsPerThreadblock[1]
  
          # calculate unroll
          # ensure that every iteration at least a full load of A,B are done
          unrollMin = 8
          unrollMin0 = totalThreads // threadblockTile[0]
          unrollMin1 = totalThreads // threadblockTile[1]
          unroll = max(unrollMin, unrollMin0, unrollMin1)
  
          threadTileM = warpShape[0] // warpThreadsM
          threadTileN = warpShape[1] // warpThreadsN
          if threadTileM < 2 or threadTileN < 2: continue
          if threadTileM*threadTileN*precisionBits > 8*8*32: continue
  
          # epilogue currently only supports N < WarpNumThreads
          if threadblockTile[1] < warpNumThreads: continue
  
          # limit smem
          smemBitsA = threadblockTile[0]*unroll*2*precisionBits
          smemBitsB = threadblockTile[1]*unroll*2*precisionBits
          smemKBytes = (smemBitsA+smemBitsB)/8/1024
          if (smemKBytes > 48): continue
  
          tile = TileDescription([threadblockTile[0], threadblockTile[1], unroll], \
                                 2, \
                                 [threadblockTile[0]//warpShape[0], threadblockTile[1]//warpShape[1], 1], \
                                 math_inst, min_cc, max_cc)
          
          def filter(t: TileDescription) -> bool:
            nonlocal tile
            return t.threadblock_shape[0] == tile.threadblock_shape[0] and \
                   t.threadblock_shape[1] == tile.threadblock_shape[1] and \
                   t.threadblock_shape[2] == tile.threadblock_shape[2] and \
                   t.warp_count[0] == tile.warp_count[0] and \
                   t.warp_count[1] == tile.warp_count[1] and \
                   t.warp_count[2] == tile.warp_count[2] and \
                   t.stages == tile.stages
          if not any(t for t in tile_descriptions if filter(t)): continue

          operations += GeneratesGemm(tile, data_type, layout[0], layout[1], layout[2], min_cc)
  return operations

#
def GenerateGemv_Simt(args):
  threadBlockShape_N = [128, 64, 32]
  ldgBits_A = [128, 64, 32]
  ldgBits_B = [128, 64, 32]

  layouts = [
    (LayoutType.RowMajor, LayoutType.RowMajor, LayoutType.RowMajor), 
  ]

  math_instructions = [
    MathInstruction(                                  \
      [1, 1, 1],                                      \
      DataType.f32, DataType.f32, DataType.f32,       \
      OpcodeClass.Simt,                               \
      MathOperation.multiply_add),
  ]
 
  min_cc = 50

  operations = []
  for math_inst in math_instructions:
    for layout in layouts:
      data_type = [
        math_inst.element_a,
        math_inst.element_b,
        math_inst.element_accumulator,
        math_inst.element_accumulator,
      ]
      for threadblock_shape_n in threadBlockShape_N:
        for align_a in ldgBits_A:
          for align_b in ldgBits_B:
            ldg_elements_a = align_a // DataTypeSize[math_inst.element_a]
            ldg_elements_b = align_b // DataTypeSize[math_inst.element_b]
            threadblock_shape_k = (256 * ldg_elements_a) // (threadblock_shape_n // ldg_elements_b)
            threadblock_shape = [1, threadblock_shape_n, threadblock_shape_k]
            thread_shape = [1, ldg_elements_b, ldg_elements_a]

            operations.append(GeneratesGemv(math_inst, \
                                            threadblock_shape, \
                                            thread_shape, \
                                            data_type, \
                                            layout[0], \
                                            layout[1], \
                                            layout[2], \
                                            min_cc, \
                                            align_a, \
                                            align_b))
  return operations

#
def GenerateConv2dOperations(args):
  if args.type == "simt":
    return GenerateConv2d_Simt(args)
  elif args.type == "tensorop8816":
    return GenerateConv2d_TensorOp_8816(args)
  else:
    assert args.type == "tensorop8832", "operation conv2d only support" \
        "simt, tensorop8816 and tensorop8832. (got:{})".format(args.type)
    return GenerateConv2d_TensorOp_8832(args)

def GenerateDeconvOperations(args):
  assert args.type == "simt", "operation deconv only support" \
        "simt. (got:{})".format(args.type)
  return GenerateDeconv_Simt(args)

def GenerateGemmOperations(args):
  assert args.type == "simt", "operation gemm only support" \
        "simt. (got:{})".format(args.type)
  return GenerateGemm_Simt(args)

def GenerateGemvOperations(args):
  assert args.type == "simt", "operation gemv only support" \
        "simt. (got:{})".format(args.type)
  return GenerateGemv_Simt(args)

###################################################################################################
###################################################################################################

if __name__ == "__main__":

  parser = argparse.ArgumentParser(description="Generates device kernel registration code for CUTLASS Kernels")
  parser.add_argument("--operations", type=str, choices=['gemm', 'gemv', 'conv2d', 'deconv'], 
                      required=True, help="Specifies the operation to generate (gemm, gemv, conv2d, deconv)")
  parser.add_argument("output", type=str, help="output directory for CUTLASS kernel files")
  parser.add_argument("--type", type=str, choices=['simt', 'tensorop8816', 'tensorop8832'], 
                      default='simt', help="kernel type of CUTLASS kernel generator")

  operation2wrapper_path = {
    "gemm": "src/cuda/matrix_mul/cutlass_matrix_mul_wrapper.cuinl", \
    "gemv": "src/cuda/matrix_mul/cutlass_matrix_mul_wrapper_batched_gemv_strided.cuinl", \
    "conv2d": "src/cuda/conv_bias/implicit_gemm_conv_bias_cutlass_wrapper.cuinl", \
    "deconv": "src/cuda/convolution/backward_data/implicit_gemm_deconv_cutlass_wrapper.cuinl", \
  }

  args = parser.parse_args()

  wrapper_path = operation2wrapper_path[args.operations] 
  if args.operations == "gemm":
    operations = GenerateGemmOperations(args)
  elif args.operations == "gemv":
    operations = GenerateGemvOperations(args)
  elif args.operations == "conv2d":
    operations = GenerateConv2dOperations(args)
  elif args.operations == "deconv":
    operations = GenerateDeconvOperations(args)
  

  if args.operations == "conv2d" or args.operations == "deconv":
    for operation in operations:
      with EmitConvSingleKernelWrapper(args.output, operation, wrapper_path) as emitter:
        emitter.emit()
  elif args.operations == "gemm" or args.operations == "gemv":
    for operation in operations:
      with EmitGemmSingleKernelWrapper(args.output, operation, wrapper_path) as emitter:
        emitter.emit()
  
#
###################################################################################################
