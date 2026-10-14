class PIDController:

    def __init__(self,kp,ki,kd):
        self.kp=kp
        self.ki=ki
        self.kd=kd

        self.previous_error=0
        self.integral=0
        self.was_line_lost=False
        self.max_integral = 10
        self.min_integral = -10
        self.max_correction=100
        self.min_correction=-100
        self.filtered_error = 0
        self.alpha = 0.2
        self.initialized=False


    def update(self,error,dt,line_detected=True):
        if not(line_detected):
            self.integral=0
            self.previous_error=0
            self.filtered_error = 0
            self.was_line_lost=True
            self.initialized=False
            return 0

        if(self.was_line_lost):
            self.previous_error=error
            self.was_line_lost=False

        if not self.initialized:
            self.filtered_error=error
            self.initialized=True

        else:
            self.filtered_error=self.alpha*error + (1- self.alpha)*self.filtered_error

        error=self.filtered_error

        if dt<=0:
            return 0
        P=self.kp*error
        D=((error - self.previous_error)/dt)*self.kd

        samp_integral=self.integral + error*dt
        samp_integral=max(self.min_integral,min(samp_integral,self.max_integral))

        samp_I=self.ki*samp_integral
        samp_correction=P+D+samp_I

        pushing_high=(samp_correction>self.max_correction and error>0)
        pushing_low=(samp_correction<self.min_correction and error<0)

        if not(pushing_high or pushing_low):
            self.integral=samp_integral

        I=self.ki*self.integral

        self.previous_error=error
        correction=P+I+D
        output=max(self.min_correction,min(correction,self.max_correction))

        return output

    
def calculate_motion(correction, base_speed=0.15,line_detected=True):
        if not line_detected:
            return 0.0, 0.0

        max_angular_speed=1.5

        angular_speed=max(-max_angular_speed,min(max_angular_speed,correction))

        turn_ratio=abs(angular_speed)/max_angular_speed

        linear_speed=base_speed*(1-0.5*turn_ratio)

        return linear_speed,angular_speed

    


