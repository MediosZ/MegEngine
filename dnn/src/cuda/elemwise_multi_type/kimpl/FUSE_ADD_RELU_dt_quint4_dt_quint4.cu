// generated by gen_elemwise_multi_type_kern_impls.py
#define KERN_IMPL_MODE(cb) MEGDNN_ELEMWISE_MODE_ENABLE(FUSE_ADD_RELU, cb)
#define KERN_IMPL_ARITY 2
#define KERN_IMPL_STYPE dt_quint4
#define KERN_IMPL_DTYPE dt_quint4
#include "../kern_impl_q4.inl"
