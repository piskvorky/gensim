/**
 * Parallax 
 * 
 * Translate3d 
 * 
 * 1.0 | Muffin Group
 */

mfnSetup = {
    translate: null
};


(function( $ ){
    "use strict";
    
    /* ------------------------------------------------------------------------
     * 
     * mfnSetup
     *
     * --------------------------------------------------------------------- */
    
    
    // has3d ------------------------------------------------
    
    var has3d = function(){
    	
        if ( ! window.getComputedStyle ) {
            return false;
        }

        var el = document.createElement('div'), 
            has3d,
            transforms = {
                'webkitTransform':'-webkit-transform',
                'OTransform':'-o-transform',
                'msTransform':'-ms-transform',
                'MozTransform':'-moz-transform',
                'transform':'transform'
            };

        document.body.insertBefore(el, null);

        for (var t in transforms) {
            if (el.style[t] !== undefined) {
                el.style[t] = "translate3d(1px,1px,1px)";
                has3d = window.getComputedStyle(el).getPropertyValue(transforms[t]);
            }
        }

        document.body.removeChild(el);

        return( has3d !== undefined && has3d !== null && has3d.length > 0 && has3d !== "none" );
    }

    
    // browserPrefix ------------------------------------------------
    
    var browserPrefix = function () {
    	
        var el = document.createElement('div'),
        	vendor = ["ms", "O", "Webkit", "Moz"],
        	i, prefix;
        
        for (i in vendor) {
            if (el.style[vendor[i] + "Transition"] !== undefined) {
            	prefix = vendor[i];
                break;
            }
        }
        return prefix;
    };


    // __construct ------------------------------------------------
    
    var __construct = function () {

        if( has3d() ){
        	
        	mfnSetup.translate = function (el, x, y) {
                el.css( '-' + browserPrefix() + '-transform', 'translate3d(' + x + ', ' + y + ', 0)' );
            };
            
        } else {
        	
            mfnSetup.translate = function (el, x, y) {
                el.css({ "left": x, "top": y });
            };
            
        }
    };

    __construct();

})(jQuery);


(function( $ ){
    "use strict";
    
    /* ------------------------------------------------------------------------
     * 
     * $.fn.mfnParallax
     *
     * --------------------------------------------------------------------- */
 
    $.fn.mfnParallax = function () {
    	
        var el = $(this),
        	parent = el.parent(),
        	speed = 500,
        	element, parentPos, windowH, translate;

        
        // imageSize ------------------------------------------------
        
        var imageSize = function( img ){
        	
        	var w, h, l, t;	// width, height, left, top
        	
        	var imageW = img.get(0).naturalWidth;
        	var imageH = img.get(0).naturalHeight;
        	
        	var parentW = img.parent().outerWidth();
        	var parentH = img.parent().outerHeight();
        	
        	var windowH = $(window).height()
        	
        	// fix for small sections
        	if( windowH > parentH ){
        		parentH = windowH;
        	}
        	
        	var diff = imageW / parentW;
        	
        	if( (imageH / diff) < parentH ){
        		
        		w = imageW / (imageH / parentH);
        		h = parentH;
        		
        		if( w > imageW ){
        			
        			if( h > windowH ){
        				
        				w = imageW / (imageH / windowH);
        				h = windowH;
        				
        			} else {
        				
        				w = imageW;
            			h = imageH;
            			
        			}

        		}
        		
        	} else {
        		
        		w = parentW;
        		h = imageH / diff;
        		
        	}
        	
        	l = ( parentW - w ) / 2;
            t = ( parentH - h ) / 2;
            
            return [w, h, l, t];    	
        }
        
        
        // parallax ------------------------------------------------
        
        var parallax = function(){
        	
        	var scrollTop = $(window).scrollTop(),
        		scrollDiff, ratio, translateTop;

            if( parentPos !== undefined ){

                if( scrollTop >= parentPos.min && scrollTop <= parentPos.max ) {
                	
                    scrollDiff = scrollTop - parentPos.min;
                    ratio = scrollDiff / parentPos.height;
   
                    translateTop = windowH + ( ratio * speed ) - scrollDiff - ( speed * ( windowH / parentPos.height ) ) ;

                    mfnSetup.translate( el, element.left + "px", -Math.round( translateTop ) + "px" );
                }
                
            }
        };
        
        
        // init ------------------------------------------------
        
        var init = function(){

        	windowH = $(window).height();
        	

            var initElement = function () {

                var size = imageSize( el );
                
                el.removeAttr('style').css({
                	'width'		: size[0],
                    'height'	: size[1]
                });

                mfnSetup.translate(el, size[2] + "px", size[3] + "px");
                
                return {
                	'width'		: size[0],
                    'height'	: size[1],
                    'left'		: size[2],
                    'top'		: size[3]
                };
            };

            element = initElement();
            
            
            var initParent = function () {
            	
                var min = parent.offset().top - $(window).height();
                var max = parent.offset().top + $(parent).outerHeight();

                return {
                    'min'		: min,
                    'max'		: max,
                    'height'	: max - min
                };
            };
            
            parentPos = initParent();
        };

        
        // reload ------------------------------------------------
        
        var reload = function () {
        	
            setTimeout(function () {
                init();
                parallax();
            }, 200);
            
        };
        
        
        // .bind() ------------------------------------------------
        
        $(window).bind('load resize', reload);
        $(window).bind('scroll', parallax);

    };   
    

    /* ------------------------------------------------------------------------
     * 
     * $(document).ready
     *
     * --------------------------------------------------------------------- */
    
    $(document).ready(function() {
		
		if( $(".mfn-parallax").length ) {
        	
            $(".mfn-parallax").each( function(){
            	$(this).mfnParallax();
            });

        }
		
	});
    
    
})(jQuery);