struct ConcaveVexCalibrationInputParams /// \brief json
{
    std::string group = dv_key_values::k_concave_vex_calib_params_key;  /// \brief group
    dv_param::DVInt detection_grayscale_thresh ;                        /// \brief 识别灰度
    dv_param::DVInt detection_area_thresh = 10000;                      /// \brief 识别面面积
    dv_param::DVRect roi_border = dv_param::DVRect(0, 0, 1440, 1440);   /// \brief 识别上下左右边
    dv_param::DVInt qualified_grayscale_thresh = 100;                   /// \brief 合格灰度
}