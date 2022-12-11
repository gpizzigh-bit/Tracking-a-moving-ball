from my_kf import KF
import unittest
import pytest
import numpy as np


class Test_KF(unittest.TestCase):
    def test_can_construct_with_x_and_v(self):
        x = 0.2
        v = 0.5

        kf = KF(init_x = x, init_v= v,accel_var=1.2)
        self.assertAlmostEqual(kf.pos, x)
        self.assertAlmostEqual(kf.vel, v)
    
    def test_if_predict_is_of_right_shape(self):
        x = 0.2
        v = 0.5

        kf = KF(init_x = x, init_v= v,accel_var=1.2)
        kf.predict(dt = 0.1)

        self.assertEqual(kf.cov.shape, (2,2))
        self.assertEqual(kf.mean.shape, (2,))
    
    def test_if_increases_state_uncertainty(self):
        x = 0.2
        v = 0.5

        kf = KF(init_x = x, init_v= v,accel_var=1.2)
        for i in range(10):
            old_det = np.linalg.det(kf.cov)
            kf.predict(dt = 0.1)
            new_det = np.linalg.det(kf.cov)

            self.assertGreater(new_det, old_det)
            print(f"inc -> [new:{new_det}, old:{old_det}]")
        
    # [TODO] [OPTIONAL] - assert the numerical values of the state before and after the predict

    def test_if_decreases_state_uncertainty(self):
        x = 0.2
        v = 0.5

        kf = KF(init_x = x, init_v= v,accel_var=1.2)
        for i in range(10):
            old_det = np.linalg.det(kf.cov)
            kf.predict(dt = 0.1)
            kf.update(meas_value=10.0,meas_variance=0.1)
            new_det = np.linalg.det(kf.cov)
            
            self.assertLess(new_det, old_det)
            print(f"dec -> [new:{new_det}, old:{old_det}]")






