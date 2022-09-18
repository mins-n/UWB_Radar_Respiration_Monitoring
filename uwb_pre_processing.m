classdef uwb_pre_processing
    properties
        %rader default bandwith range(단위 : ghz)
        band_min = 6e+9;
        band_max = 8.5e+9;
        %rader default sampling rate(1536 samples(row index) per sec
        sampling_rate = 23.328e+9;
        PRF = 15.1875e+6; %pulse repetition frequency
        PRI = 1/PRF;
        c=299792458;%light speed


        pulse_per_step
        iteration

        fps = 20;
        fftsize = 256;
        v_map=[];
    end
    methods
        function obj=process(obj, rawdata)
            Y=fft(rawdata,[],2);
            phase_Y=unwrap(angle(Y));
            Z=fft(exp(-1j*phase_Y),[],2);
            obj.v_map=max(abs(Z))*obj.c/(2*obj.band_min);
        end
    end
end