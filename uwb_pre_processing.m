classdef uwb_pre_processing
    properties
        %rader default bandwith range(단위 : ghz)
        band_min = 6e+9;
        band_max = 8.5e+9;
        %rader default sampling rate(1536 samples(row index) per sec
        sampling_rate = 23.328e+9;
        %pulse repetition frequency
        PRF = 15.1875e+6;

        %light speed
        c=299792458;


        pulse_per_step
        iteration

        fps = 20;
        fftsize = 256;

        v_map=[];
    end
    methods
        function obj=process(obj, rawdata)
            fft_data=fft(rawdata,[],2);
            fft_phase_data=unwrap(angle(fft_data),[],2);
            velocity=fft(exp(-1j*fft_phase_data),[],2);
            obj.v_map = max(abs(velocity),[],2)*obj.c/(2*obj.band_min);
        end
    end
end